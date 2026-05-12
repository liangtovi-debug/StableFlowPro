import Foundation
import AVFoundation
import CoreVideo
import Metal
import MetalKit
import simd

class GPUStabilizer {
    
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var textureCache: CVMetalTextureCache?
    
    var opticalFlowPipeline: MTLComputePipelineState?
    var opticalFlowMultiPointPipeline: MTLComputePipelineState?  // 并行多点光流
    var templateMatchPipeline: MTLComputePipelineState?
    var extractPatchPipeline: MTLComputePipelineState?
    var renderPipeline: MTLRenderPipelineState?
    
    // Track state
    var currentPoint: SIMD2<Float> = .zero
    var lastGoodPoint: SIMD2<Float> = .zero
    var velocity: SIMD2<Float> = .zero
    
    // 模板匹配相关
    var templateTexture: MTLTexture?
    var templatePatchSize: Int = 21
    var searchRadiusTemplate: Float = 40  // 模板搜索范围
    var frameIndex: Int = 0
    
    // 融合配置
    var flowFailThreshold: Float = 24.0    // 光流跳变阈值
    var stableConfidenceThreshold: Float = 0.005  // 稳定判定阈值
    var stableMotionThreshold: Float = 20.0       // 稳定时的运动阈值
    
    // 动态搜索半径
    var minSearchRadius: Float = 20.0
    var maxSearchRadius: Float = 120.0
    var templateRefreshInterval: Int = 12
    var templateValidationDistance: Float = 0.035
    
    // Multi-point tracking (3x3 grid = 9 points)
    let pointOffsets: [SIMD2<Float>] = [
        SIMD2<Float>(-6, -6), SIMD2<Float>(0, -6), SIMD2<Float>(6, -6),
        SIMD2<Float>(-6, 0),  SIMD2<Float>(0, 0),  SIMD2<Float>(6, 0),
        SIMD2<Float>(-6, 6),  SIMD2<Float>(0, 6),  SIMD2<Float>(6, 6)
    ]
    
    // 配置
    var searchRadius: Int = 16  // 增大搜索范围
    var confidenceThreshold: Float = 0.15
    
    // 调试模式
    var debugMode: Bool = false
    
    init?() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal not supported")
            return nil
        }
        self.device = device
        
        guard let queue = device.makeCommandQueue() else {
            return nil
        }
        self.commandQueue = queue
        
        var cache: CVMetalTextureCache?
        let status = CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        if status != kCVReturnSuccess {
            print("⚠️ Metal 纹理缓存创建失败: \(status)")
        }
        self.textureCache = cache
        
        if debugMode {
            print("GPUStabilizer 初始化完成")
            print("  - opticalFlowPipeline: \(opticalFlowPipeline != nil)")
            print("  - opticalFlowMultiPointPipeline: \(opticalFlowMultiPointPipeline != nil)")
            print("  - textureCache: \(textureCache != nil)")
        }
        
        setupPipelines()
    }
    
    private func setupPipelines() {
        guard let library = loadMetalLibrary() else {
            print("Failed to load Metal library")
            return
        }
        
        // Gradient-based optical flow
        if let func_ = library.makeFunction(name: "opticalFlowGradient") {
            opticalFlowPipeline = try? device.makeComputePipelineState(function: func_)
        }
        
        // 并行多点光流 - 一次dispatch处理9个点
        if let func_ = library.makeFunction(name: "opticalFlowMultiPoint") {
            opticalFlowMultiPointPipeline = try? device.makeComputePipelineState(function: func_)
        }
        
        // Template matching pipeline
        if let func_ = library.makeFunction(name: "templateMatchParallel") {
            templateMatchPipeline = try? device.makeComputePipelineState(function: func_)
        }
        
        // Extract patch pipeline
        if let func_ = library.makeFunction(name: "extractPatch") {
            extractPatchPipeline = try? device.makeComputePipelineState(function: func_)
        }
        
        // Render pipeline
        if let vs = library.makeFunction(name: "stableVertex"),
           let fs = library.makeFunction(name: "stableFragment") {
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.vertexFunction = vs
            descriptor.fragmentFunction = fs
            descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            
            renderPipeline = try? device.makeRenderPipelineState(descriptor: descriptor)
        }
    }

    private func loadMetalLibrary() -> MTLLibrary? {
        if let defaultLibrary = device.makeDefaultLibrary() {
            return defaultLibrary
        }

        let candidateURLs = metalSourceCandidateURLs()
        for url in candidateURLs where FileManager.default.fileExists(atPath: url.path) {
            if let source = try? String(contentsOf: url, encoding: .utf8),
               let library = try? device.makeLibrary(source: source, options: nil) {
                if debugMode {
                    print("GPUStabilizer: Loaded Metal source from \(url.path)")
                }
                return library
            }
        }

        return nil
    }

    private func metalSourceCandidateURLs() -> [URL] {
        var urls: [URL] = []

        // 安全查找 SPM resource bundle（避免 Bundle.module 的 fatalError）
        let bundleName = "StableFlow_StableFlow"
        let candidateBundles: [Bundle?] = [
            Bundle.main.url(forResource: bundleName, withExtension: "bundle")
                .flatMap { Bundle(url: $0) },
            {
                guard let resURL = Bundle.main.resourceURL else { return nil }
                return Bundle(url: resURL.appendingPathComponent("\(bundleName).bundle"))
            }(),
        ]
        for case let bundle? in candidateBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: "metal") {
                urls.append(url)
            }
        }

        // 注意：不使用 Bundle.module，因为它在非开发环境会 fatalError

        if let bundled = Bundle.main.url(forResource: "Shaders", withExtension: "metal") {
            urls.append(bundled)
        }

        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        urls.append(cwd.appendingPathComponent("Sources/Shaders.metal"))

        if let executableDir = Bundle.main.executableURL?.deletingLastPathComponent() {
            urls.append(executableDir.appendingPathComponent("Shaders.metal"))
            urls.append(executableDir.appendingPathComponent("../Resources/Shaders.metal").standardizedFileURL)
            // app bundle 内的 resource bundle 路径
            urls.append(executableDir
                .appendingPathComponent("../Resources/\(bundleName).bundle/Shaders.metal")
                .standardizedFileURL)
        }

        return urls
    }
    
    func createTexture(from pixelBuffer: CVPixelBuffer) -> MTLTexture? {
        guard let cache = textureCache else { 
            if debugMode { print("错误: textureCache 为 nil") }
            return nil 
        }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            nil, cache, pixelBuffer, nil,
            .bgra8Unorm, width, height, 0, &cvTexture
        )
        
        guard status == kCVReturnSuccess, let texture = cvTexture else {
            if debugMode { print("错误: 纹理创建失败 status=\(status)") }
            return nil
        }
        
        return CVMetalTextureGetTexture(texture)
    }
    
    // 跟踪单点（带梯度特征 + 置信度）
    func trackPointWithGradient(
        prev: MTLTexture,
        curr: MTLTexture,
        point: SIMD2<Float>
    ) -> (offset: SIMD2<Float>, confidence: Float)? {
        guard let pipeline = opticalFlowPipeline else { return nil }
        
        let width = Float(prev.width)
        let height = Float(prev.height)
        
        guard let resultX = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared),
              let resultY = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared),
              let resultCount = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared) else {
            return nil
        }
        
        // 初始化
        var zero: UInt32 = 0
        memcpy(resultX.contents(), &zero, MemoryLayout<UInt32>.size)
        memcpy(resultY.contents(), &zero, MemoryLayout<UInt32>.size)
        memcpy(resultCount.contents(), &zero, MemoryLayout<UInt32>.size)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(prev, index: 0)
        encoder.setTexture(curr, index: 1)
        encoder.setBuffer(resultX, offset: 0, index: 0)
        encoder.setBuffer(resultY, offset: 0, index: 1)
        encoder.setBuffer(resultCount, offset: 0, index: 2)
        
        var pt = point
        var size = SIMD2<Float>(width, height)
        var radius = Int32(searchRadius)
        
        encoder.setBytes(&pt, length: MemoryLayout<SIMD2<Float>>.size, index: 3)
        encoder.setBytes(&size, length: MemoryLayout<SIMD2<Float>>.size, index: 4)
        encoder.setBytes(&radius, length: MemoryLayout<Int32>.size, index: 5)
        
        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let numGroups = MTLSize(
            width: (prev.width + 15) / 16,
            height: (prev.height + 15) / 16,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // 读取结果
        let dx = Int32(bitPattern: resultX.contents().load(as: UInt32.self))
        let dy = Int32(bitPattern: resultY.contents().load(as: UInt32.self))
        let count = Int32(bitPattern: resultCount.contents().load(as: UInt32.self))
        
        // 如果没有足够多线程匹配成功，降低置信度
        let confidence = Float(count) / Float(numGroups.width * numGroups.height * 4)
        
        guard count > 0 else {
            return nil  // 跟踪失败
        }
        
        let avgDx = Float(dx) / Float(count)
        let avgDy = Float(dy) / Float(count)
        
        return (SIMD2<Float>(avgDx, avgDy), confidence)
    }
    
    // 多点投票跟踪
    func trackPointMultiPoint(
        prev: MTLTexture,
        curr: MTLTexture,
        centerPoint: SIMD2<Float>
    ) -> SIMD2<Float>? {
        
        var votes: [SIMD2<Float>] = []
        var totalConfidence: Float = 0
        
        // 对9个点进行跟踪
        for offset in pointOffsets {
            let point = SIMD2<Float>(
                centerPoint.x + offset.x / Float(prev.width),
                centerPoint.y + offset.y / Float(prev.height)
            )
            
            // 边界检查
            if point.x < 0.02 || point.x > 0.98 || point.y < 0.02 || point.y > 0.98 {
                continue
            }
            
            if let result = trackPointWithGradient(prev: prev, curr: curr, point: point) {
                // 置信度高才接受
                if result.confidence > confidenceThreshold {
                    votes.append(result.offset)
                    totalConfidence += result.confidence
                }
            }
        }
        
        // 需要至少3个点成功才认为是稳定的
        guard votes.count >= 3 else {
            return nil
        }
        
        // 加权平均
        var avgX: Float = 0
        var avgY: Float = 0
        
        for vote in votes {
            avgX += vote.x
            avgY += vote.y
        }
        
        avgX /= Float(votes.count)
        avgY /= Float(votes.count)
        
        // 限制最大跳变
        let maxJump: Float = 20.0
        avgX = max(-maxJump, min(maxJump, avgX))
        avgY = max(-maxJump, min(maxJump, avgY))
        
        return SIMD2<Float>(avgX, avgY)
    }
    
    // 🚀 并行多点光流 - 一次GPU dispatch处理9个点（大幅提速）
    func trackPointMultiPointParallel(
        prev: MTLTexture,
        curr: MTLTexture,
        centerPoint: SIMD2<Float>
    ) -> SIMD2<Float>? {
        
        guard let pipeline = opticalFlowMultiPointPipeline else {
            // fallback to old method
            return trackPointMultiPoint(prev: prev, curr: curr, centerPoint: centerPoint)
        }
        
        let width = Float(prev.width)
        let height = Float(prev.height)
        
        // 准备9个点的坐标（归一化）
        var points: [SIMD2<Float>] = []
        for offset in pointOffsets {
            let px = centerPoint.x + offset.x / Float(prev.width)
            let py = centerPoint.y + offset.y / Float(prev.height)
            
            // 边界检查
            if px > 0.02 && px < 0.98 && py > 0.02 && py < 0.98 {
                points.append(SIMD2<Float>(px, py))
            } else {
                points.append(centerPoint)  // 用中心点代替
            }
        }
        
        // 创建结果 buffer
        guard let resultX = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared),
              let resultY = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared),
              let resultCount = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared) else {
            return nil
        }
        
        // 初始化
        var zero: UInt32 = 0
        memcpy(resultX.contents(), &zero, MemoryLayout<UInt32>.size)
        memcpy(resultY.contents(), &zero, MemoryLayout<UInt32>.size)
        memcpy(resultCount.contents(), &zero, MemoryLayout<UInt32>.size)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(prev, index: 0)
        encoder.setTexture(curr, index: 1)
        encoder.setBuffer(resultX, offset: 0, index: 0)
        encoder.setBuffer(resultY, offset: 0, index: 1)
        encoder.setBuffer(resultCount, offset: 0, index: 2)
        
        // 传递9个点
        var pointsArray = points
        var size = SIMD2<Float>(width, height)
        var radius = Int32(searchRadius)
        var numPoints = Int32(points.count)
        
        encoder.setBytes(&pointsArray, length: MemoryLayout<SIMD2<Float>>.size * points.count, index: 3)
        encoder.setBytes(&size, length: MemoryLayout<SIMD2<Float>>.size, index: 4)
        encoder.setBytes(&radius, length: MemoryLayout<Int32>.size, index: 5)
        encoder.setBytes(&numPoints, length: MemoryLayout<Int32>.size, index: 6)
        
        // 一次 dispatch 处理所有点
        let threadsPerGroup = MTLSize(width: 9, height: 1, depth: 1)
        let numGroups = MTLSize(width: 1, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // 读取结果
        let dx = Int32(bitPattern: resultX.contents().load(as: UInt32.self))
        let dy = Int32(bitPattern: resultY.contents().load(as: UInt32.self))
        let count = Int32(bitPattern: resultCount.contents().load(as: UInt32.self))
        
        guard count > 0 else {
            return nil  // 跟踪失败
        }
        
        let avgDx = Float(dx) / Float(count)
        let avgDy = Float(dy) / Float(count)
        
        // 限制最大跳变
        let maxJump: Float = 20.0
        let clampedDx = max(-maxJump, min(maxJump, avgDx))
        let clampedDy = max(-maxJump, min(maxJump, avgDy))
        
        return SIMD2<Float>(clampedDx, clampedDy)
    }
    
    // 主跟踪函数
    func track(
        prev: MTLTexture,
        curr: MTLTexture,
        centerPoint: SIMD2<Float>
    ) -> SIMD2<Float> {
        
        // 使用多点投票
        if let offset = trackPointMultiPoint(prev: prev, curr: curr, centerPoint: centerPoint) {
            
            // 计算新位置
            let newX = centerPoint.x + offset.x / Float(curr.width)
            let newY = centerPoint.y + offset.y / Float(curr.height)
            
            // 惯性平滑 (0.7 old + 0.3 new)
            currentPoint = SIMD2<Float>(
                currentPoint.x * 0.7 + newX * 0.3,
                currentPoint.y * 0.7 + newY * 0.3
            )
            
            velocity = offset
            lastGoodPoint = currentPoint
        } else {
            // 跟踪失败，保持位置并衰减速度
            velocity = velocity * 0.5
        }
        
        return currentPoint
    }
    
    // 渲染
    func applyTransform(input: MTLTexture, output: MTLTexture, offset: SIMD2<Float>, scale: Float) {
        guard let pipeline = renderPipeline,
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }
        
        let descriptor = MTLRenderPassDescriptor()
        descriptor.colorAttachments[0].texture = output
        descriptor.colorAttachments[0].loadAction = .clear
        descriptor.colorAttachments[0].storeAction = .store
        descriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else {
            return
        }
        
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentTexture(input, index: 0)
        
        var off = offset
        var sc = scale
        encoder.setFragmentBytes(&off, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
        encoder.setFragmentBytes(&sc, length: MemoryLayout<Float>.size, index: 1)
        
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    // ============================================================
    // 模板匹配相关功能
    // ============================================================
    
    // 初始化模板（从当前帧提取patch）
    func initTemplate(from texture: MTLTexture, at point: SIMD2<Float>) {
        let size = templatePatchSize
        
        // 创建模板纹理
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm,
            width: size,
            height: size,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        
        guard let patchTex = device.makeTexture(descriptor: descriptor) else {
            print("Failed to create patch texture")
            return
        }

        guard let pipeline = extractPatchPipeline,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setTexture(patchTex, index: 1)
        
        var center = point
        var imageSize = SIMD2<Float>(Float(texture.width), Float(texture.height))
        
        encoder.setBytes(&center, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
        encoder.setBytes(&imageSize, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
        
        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let numGroups = MTLSize(
            width: (size + 15) / 16,
            height: (size + 15) / 16,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        self.templateTexture = patchTex
        print("Template initialized at \(point)")
    }
    
    // 更新模板（每隔N帧）
    func updateTemplate(from texture: MTLTexture, at point: SIMD2<Float>) {
        initTemplate(from: texture, at: point)
    }
    
    // 模板匹配搜索（全局搜索）
    func templateMatch(texture: MTLTexture) -> (position: SIMD2<Float>, error: Float)? {
        guard let templateTex = templateTexture,
              let pipeline = templateMatchPipeline else {
            return nil
        }
        
        let searchR = Int(searchRadiusTemplate)
        let totalCandidates = (searchR * 2 + 1) * (searchR * 2 + 1)
        
        // 创建误差buffer
        guard let errorsBuffer = device.makeBuffer(
            length: totalCandidates * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            return nil
        }
        
        // 初始化误差为最大值
        let initErrors = [Float](repeating: 1e9, count: totalCandidates)
        memcpy(errorsBuffer.contents(), initErrors, totalCandidates * MemoryLayout<Float>.size)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setTexture(templateTex, index: 1)
        encoder.setBuffer(errorsBuffer, offset: 0, index: 0)
        
        var center = currentPoint
        var imageSize = SIMD2<Float>(Float(texture.width), Float(texture.height))
        var radius = searchRadiusTemplate
        
        encoder.setBytes(&center, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
        encoder.setBytes(&imageSize, length: MemoryLayout<SIMD2<Float>>.size, index: 2)
        encoder.setBytes(&radius, length: MemoryLayout<Float>.size, index: 3)
        
        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let numGroups = MTLSize(
            width: (searchR * 2 + 15) / 16,
            height: (searchR * 2 + 15) / 16,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // 找最小误差
        let errors = errorsBuffer.contents().bindMemory(to: Float.self, capacity: totalCandidates)
        
        var minError: Float = 1e9
        var minIndex: Int = 0
        
        for i in 0..<totalCandidates {
            if errors[i] < minError {
                minError = errors[i]
                minIndex = i
            }
        }
        
        // 计算最佳位置
        let dx = minIndex % (searchR * 2 + 1) - searchR
        let dy = minIndex / (searchR * 2 + 1) - searchR
        
        let bestPosition = SIMD2<Float>(
            currentPoint.x + Float(dx) / Float(texture.width),
            currentPoint.y + Float(dy) / Float(texture.height)
        )
        
        return (bestPosition, minError)
    }
    
    // ============================================================
    // 融合跟踪（光流 + 模板匹配）
    // ============================================================
    
    // 带重定位的跟踪
    func trackWithRecovery(
        prev: MTLTexture,
        curr: MTLTexture,
        centerPoint: SIMD2<Float>
    ) -> SIMD2<Float> {
        
        frameIndex += 1
        
        // 第一次初始化模板
        if templateTexture == nil {
            initTemplate(from: prev, at: centerPoint)
            currentPoint = centerPoint
            lastGoodPoint = centerPoint
            return currentPoint
        }
        
        // 1. 光流跟踪
        // 🚀 使用并行多点光流（大幅提速）
        let flowOffset = trackPointMultiPointParallel(prev: prev, curr: curr, centerPoint: centerPoint)
        
        // 2. 计算光流是否失效
        var flowValid = false
        var flowResult = centerPoint
        var motion: Float = 0
        
        if let offset = flowOffset {
            motion = hypot(offset.x, offset.y)
            
            // 检查跳变是否过大
            if motion < flowFailThreshold {
                flowValid = true
                flowResult = SIMD2<Float>(
                    centerPoint.x + offset.x / Float(curr.width),
                    centerPoint.y + offset.y / Float(curr.height)
                )
            }
        }
        
        // 3. 动态搜索半径（基于速度）
        searchRadiusTemplate = min(max(motion * 2.0, minSearchRadius), maxSearchRadius)
        
        // 4. 模板匹配（失效时 OR 每5帧验证）
        var templateResult: SIMD2<Float>?
        var needsRecovery = false
        
        if !flowValid {
            // 光流失败，强制用模板匹配
            needsRecovery = true
            templateResult = templateMatch(texture: curr)?.position
        } else if frameIndex % 5 == 0 {
            // 即使光流有效，也每隔5帧做一次模板匹配验证
            templateResult = templateMatch(texture: curr)?.position
        }
        
        // 5. 融合策略：不是加权平均，而是选择信任谁
        var finalPoint: SIMD2<Float>
        
        if needsRecovery, let tmpl = templateResult {
            // 关键恢复：用模板匹配结果强制拉回
            finalPoint = tmpl
            updateTemplate(from: curr, at: finalPoint)
            if debugMode {
                print("🔄 Recovery: flow failed (motion=\(motion)), using template match")
            }
        } else if flowValid {
            // 光流有效，信任光流
            finalPoint = flowResult

            // 只在模板验证也一致时，低频刷新模板，避免模板追着背景一起漂。
            if let tmpl = templateResult {
                let disagreement = hypot(tmpl.x - flowResult.x, tmpl.y - flowResult.y)
                let shouldRefreshTemplate =
                    disagreement < templateValidationDistance &&
                    motion < stableMotionThreshold * 0.35 &&
                    frameIndex % max(templateRefreshInterval, 1) == 0
                if shouldRefreshTemplate && templateTexture != nil {
                    updateTemplate(from: curr, at: finalPoint)
                }
            }
        } else if let tmpl = templateResult {
            // 只有模板有效
            finalPoint = tmpl
            updateTemplate(from: curr, at: finalPoint)
        } else {
            // 都失败，退回最近一次可靠位置，避免继续把漂移累积进轨迹。
            finalPoint = lastGoodPoint
            if debugMode {
                print("⚠️ Both flow and template failed!")
            }
        }
        
        // 7. 边界检查
        finalPoint.x = max(0.05, min(0.95, finalPoint.x))
        finalPoint.y = max(0.05, min(0.95, finalPoint.y))
        
        // 8. 跟踪层不要再引入明显滞后。
        // 之前这里的惯性平滑会让点位在持续运动时总慢半拍，
        // 后面的“特征居中”再怎么补偿，也是在追一个已经落后的点。
        let trackingBlend: Float
        if needsRecovery {
            trackingBlend = 0.82
        } else if flowValid {
            trackingBlend = 0.99
        } else {
            trackingBlend = 0.92
        }

        currentPoint = SIMD2<Float>(
            currentPoint.x + (finalPoint.x - currentPoint.x) * trackingBlend,
            currentPoint.y + (finalPoint.y - currentPoint.y) * trackingBlend
        )

        if flowValid || needsRecovery || templateResult != nil {
            lastGoodPoint = currentPoint
        }

        return currentPoint
    }
}
