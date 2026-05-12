import Foundation
import AVFoundation
import CoreVideo
import CoreImage
import Metal
import Vision
import Accelerate

struct StabilizationAnalysis {
    let frameSize: CGSize
    let rawTrajectory: [CGPoint]
    let smoothedTrajectory: [CGPoint]
    let renderedTrajectory: [CGPoint]
    let transforms: [CGPoint]
    let isCentering: Bool
    let cropRatio: Double
    let cropLimited: Bool
    let meanCenterError: Double
    let maxCenterError: Double
    let meanTransformDelta: Double
    let maxTransformDelta: Double
}

class ImprovedStabilizer {
    
    var gpuStabilizer: GPUStabilizer?
    private let ciContext: CIContext = {
        if let device = MTLCreateSystemDefaultDevice() {
            return CIContext(mtlDevice: device, options: [
                .cacheIntermediates: false,
                .priorityRequestLow: false
            ])
        }
        return CIContext(options: nil)
    }()
    private(set) var lastAnalysis: StabilizationAnalysis?
    
    // 调试模式
    var debugMode: Bool = false

    func reset() {
        gpuStabilizer = nil
    }
    
    func stabilizeVideo(
        inputURL: URL,
        outputURL: URL,
        trackingPoint: CGPoint?,
        trackingBoxSize: CGSize = CGSize(width: 80, height: 80),
        exportFormat: String = "mp4",
        processingQuality: String = "平衡",
        smoothFactor: Double = 0.5,
        stabilizeStrength: Double = 0.8,
        rollingShutterCompensation: Bool = true,
        clipStart: Double = 0,
        clipEnd: Double = 1,
        progressHandler: @escaping (Double, String) -> Void
    ) -> Bool {
        
        // 调试信息
        if debugMode {
            print("=== StableFlow GPU 调试 ===")
            print("输入: \(inputURL.lastPathComponent)")
            print("GPU 可用: \(MTLCreateSystemDefaultDevice() != nil)")
        }
        
        let asset = AVURLAsset(url: inputURL)

        guard let videoInfo = loadVideoInfo(for: asset) else {
            progressHandler(1.0, "错误：无法读取视频轨道")
            if debugMode { print("错误: 无视频轨道") }
            return false
        }

        let videoTrack = videoInfo.track
        let width = Int(videoInfo.naturalSize.width)
        let height = Int(videoInfo.naturalSize.height)
        let preferredTransform = videoInfo.preferredTransform
        let totalDuration = CMTimeGetSeconds(videoInfo.duration)
        let fps = Double(videoInfo.nominalFrameRate)
        
        if debugMode {
            print("视频: \(width)x\(height), \(fps)fps, \(totalDuration)s")
        }
        
        guard width > 0 && height > 0 && fps > 0 else {
            progressHandler(1.0, "错误：无效视频参数")
            return false
        }
        
        let startTime = totalDuration * clipStart
        let endTime = totalDuration * clipEnd
        let clipDuration = endTime - startTime
        let clipTimescale = videoInfo.duration.timescale > 0
            ? videoInfo.duration.timescale
            : CMTimeScale(max(600, Int32((fps * 1000).rounded(.toNearestOrEven))))
        let clipStartTime = CMTime(seconds: startTime, preferredTimescale: clipTimescale)
        let clipDurationTime = CMTime(seconds: max(clipDuration, 0), preferredTimescale: clipTimescale)
        let clipTimeRange = CMTimeRange(start: clipStartTime, duration: clipDurationTime)
        
        // 跟踪点
        let trackX: CGFloat
        let trackY: CGFloat
        if let tp = trackingPoint {
            trackX = tp.x
            trackY = tp.y
        } else {
            trackX = CGFloat(width) / 2
            trackY = CGFloat(height) / 2
        }
        
        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        
        var trajectory: [(x: Float, y: Float)] = []
        var usedVisionTracking = false

        // 优先用 Vision 框架（CoreML GPU加速，快且精确）
        if trackingPoint != nil {
            progressHandler(0.05, "Vision 特征跟踪...")
            if let visionTrajectory = trackTrajectoryWithVision(
                asset: asset,
                videoTrack: videoTrack,
                settings: settings,
                startTime: startTime,
                endTime: endTime,
                clipDuration: clipDuration,
                fps: fps,
                initialPoint: CGPoint(x: trackX, y: trackY),
                trackingBoxSize: trackingBoxSize,
                frameSize: CGSize(width: width, height: height),
                preferredTransform: preferredTransform,
                progressHandler: progressHandler
            ) {
                trajectory = visionTrajectory.map { (x: Float($0.x), y: Float($0.y)) }
                usedVisionTracking = true
            }
        }
        
        if trajectory.isEmpty {
            // Initialize GPU stabilizer with more checks
            guard let gpu = GPUStabilizer() else {
                progressHandler(1.0, "错误：GPU 不可用")
                if debugMode { print("错误: GPUStabilizer 初始化失败") }
                return false
            }
            gpu.debugMode = debugMode
            
            // 检查 pipeline
            if gpu.opticalFlowPipeline == nil && gpu.opticalFlowMultiPointPipeline == nil {
                progressHandler(1.0, "错误：GPU Pipeline 不可用")
                if debugMode { print("错误: 所有 GPU pipeline 都为空") }
                return false
            }
            
            gpuStabilizer = gpu
            
            if debugMode { print("GPU 初始化成功，开始读取视频...") }
            
            // 初始化跟踪点（归一化坐标）
            gpu.currentPoint = SIMD2<Float>(
                Float(trackX) / Float(width),
                Float(trackY) / Float(height)
            )
            gpu.lastGoodPoint = gpu.currentPoint
            gpu.velocity = .zero
            
            progressHandler(0.05, "GPU 多点跟踪...")
            
            // ─── 第一遍：GPU 光流跟踪轨迹 ─────────────────────────────
            guard let reader = try? AVAssetReader(asset: asset) else {
                progressHandler(1.0, "错误：无法打开视频")
                return false
            }
            
            let output = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: settings)
            reader.add(output)
            reader.timeRange = clipTimeRange
            reader.startReading()
            
            var prevTexture: MTLTexture?
            var frameCount = 0
            let totalFrames = Int(clipDuration * fps)
            let trackingProgressInterval = max(1, totalFrames / 120)
            
            while let sample = output.copyNextSampleBuffer() {
                autoreleasepool {
                    let pts = CMSampleBufferGetPresentationTimeStamp(sample)
                    let ptsSec = CMTimeGetSeconds(pts)
                    
                    guard ptsSec >= startTime && ptsSec <= endTime else { return }
                    guard let pixelBuf = CMSampleBufferGetImageBuffer(sample) else { return }
                    
                    guard let currTexture = gpu.createTexture(from: pixelBuf) else { return }
                    
                    defer {
                        prevTexture = currTexture
                        frameCount += 1
                        if frameCount == 1 || frameCount == totalFrames || frameCount % trackingProgressInterval == 0 {
                            let prog = 0.05 + Double(frameCount) / Double(max(totalFrames, 1)) * 0.4
                            progressHandler(min(prog, 0.45), "GPU 跟踪中 \(frameCount)/\(totalFrames)")
                        }
                    }
                    
                    if let prev = prevTexture {
                        let trackedPoint = gpu.trackWithRecovery(prev: prev, curr: currTexture, centerPoint: gpu.currentPoint)
                        gpu.currentPoint = trackedPoint
                    }
                    
                    trajectory.append((x: gpu.currentPoint.x, y: gpu.currentPoint.y))
                }
            }
        }
        
        guard trajectory.count > 2 else {
            progressHandler(1.0, "错误：轨迹数据不足")
            return false
        }
        
        // ─── 第二步：轨迹平滑 ─────────────────────────────
        progressHandler(0.45, "平滑轨迹...")
        
        // 有显式选中的跟踪点时，目标是把该特征拉到画面中心。
        // 否则走普通防抖：只补偿原始轨迹与平滑轨迹之间的抖动量。
        let shouldCenterTrackedFeature = trackingPoint != nil
        let rawTrajectory = repairFrozenTrajectorySegments(trajectory)
        let filteredTrajectory = suppressTrajectoryOutliers(rawTrajectory)
        let centerLockTrajectory = shouldCenterTrackedFeature
            ? refineCenterLockTrajectory(
                filteredTrajectory,
                fps: fps,
                frameWidth: width,
                frameHeight: height
            )
            : filteredTrajectory
        let qualityKey = processingQuality.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let qualityWindowMultiplier: Double
        let bitrateMultiplier: Double
        switch qualityKey {
        case "快速", "fast":
            qualityWindowMultiplier = 0.85
            bitrateMultiplier = 0.8
        case "最佳", "best":
            qualityWindowMultiplier = 1.2
            bitrateMultiplier = 1.35
        default:
            qualityWindowMultiplier = 1.0
            bitrateMultiplier = 1.0
        }
        let windowSize = max(7, Int(fps * smoothFactor * 0.5 * qualityWindowMultiplier))
        let trajectoryForSmoothing = shouldCenterTrackedFeature ? centerLockTrajectory : filteredTrajectory
        let smoothTrajectory = smoothGPU(
            smoothGPU(trajectoryForSmoothing, windowSize: windowSize),
            windowSize: max(3, windowSize / 2)
        )
        
        if debugMode {
            print("=== 轨迹调试 ===")
            print("来源: \(usedVisionTracking ? "Vision" : "GPU")")
            print("轨迹点数: \(rawTrajectory.count)")
            if !rawTrajectory.isEmpty {
                print("首帧: (\(rawTrajectory[0].x), \(rawTrajectory[0].y))")
                print("末帧: (\(rawTrajectory[rawTrajectory.count-1].x), \(rawTrajectory[rawTrajectory.count-1].y))")
            }
        }
        
        // ─── 第三步：计算补偿 ─────────────────────────────
        let centerX: Float = 0.5
        let centerY: Float = 0.5
        
        var transforms: [(x: Float, y: Float)] = []
        var desiredCenterTransforms: [(x: Float, y: Float)] = []
        let centeringResponsiveness = max(0.85, min(1.0, Float(0.7 + stabilizeStrength * 0.3)))
        let centeringWarmupFrames = 0

        let compensationTrajectory = shouldCenterTrackedFeature ? centerLockTrajectory : filteredTrajectory
        for i in 0..<compensationTrajectory.count {
            let currentX = compensationTrajectory[i].x
            let currentY = compensationTrajectory[i].y
            
            if shouldCenterTrackedFeature {
                let desiredDx = (centerX - currentX) * Float(width)
                let desiredDy = (currentY - centerY) * Float(height)
                let warmupProgress: Float
                if centeringWarmupFrames > 0 {
                    let rawProgress = min(1.0, Float(i + 1) / Float(centeringWarmupFrames))
                    warmupProgress = rawProgress * rawProgress * (3 - 2 * rawProgress)
                } else {
                    warmupProgress = 1.0
                }
                let dx = desiredDx * warmupProgress
                let dy = desiredDy * warmupProgress
                desiredCenterTransforms.append((x: dx, y: dy))
                transforms.append((x: dx, y: dy))
            } else {
                let smoothX = i < smoothTrajectory.count ? smoothTrajectory[i].x : currentX
                let smoothY = i < smoothTrajectory.count ? smoothTrajectory[i].y : currentY
                let dx = (smoothX - currentX) * Float(stabilizeStrength) * Float(width)
                let dy = (currentY - smoothY) * Float(stabilizeStrength) * Float(height)
                transforms.append((x: dx, y: dy))
            }
        }

        if shouldCenterTrackedFeature, desiredCenterTransforms.count == transforms.count {
            let centeringWindowRadius = max(
                3,
                min(
                    15,
                    Int((fps * (0.06 + smoothFactor * 0.12)).rounded(.toNearestOrEven))
                )
            )
            transforms = smoothTransformSeries(
                desiredCenterTransforms,
                windowRadius: centeringWindowRadius
            )
            if debugMode {
                print("居中模式: 使用离线对称平滑补偿")
                print("居中平滑窗口: \(centeringWindowRadius)")
                print("居中响应系数: \(centeringResponsiveness)")
            }
        } else {
            // 普通防抖仍然使用单向平滑。
            let transformAlpha: Float = 0.15
            for i in 1..<transforms.count {
                transforms[i].x = transforms[i - 1].x * (1 - transformAlpha) + transforms[i].x * transformAlpha
                transforms[i].y = transforms[i - 1].y * (1 - transformAlpha) + transforms[i].y * transformAlpha
            }
        }

        // 限幅（防止突变）
        let maxStep: Float = shouldCenterTrackedFeature
            ? max(30.0, min(Float(max(width, height)) * 0.04, 80.0))
            : 10.0
        if maxStep.isFinite {
            for i in 1..<transforms.count {
                let dx = transforms[i].x - transforms[i-1].x
                let dy = transforms[i].y - transforms[i-1].y
                if abs(dx) > maxStep {
                    transforms[i].x = transforms[i-1].x + (dx > 0 ? maxStep : -maxStep)
                }
                if abs(dy) > maxStep {
                    transforms[i].y = transforms[i-1].y + (dy > 0 ? maxStep : -maxStep)
                }
            }
        }

        if rollingShutterCompensation {
            let shutterWindow = max(
                1,
                min(
                    4,
                    Int((fps * 0.012).rounded(.toNearestOrEven))
                )
            )
            transforms = smoothTransformSeries(
                transforms,
                windowRadius: shutterWindow
            )
            if debugMode {
                print("滚动快门修复: 启用")
                print("滚动快门平滑窗口: \(shutterWindow)")
            }
        } else if debugMode {
            print("滚动快门修复: 关闭")
        }
        
        if debugMode {
            print("=== GPU 变换调试 ===")
            print("变换点数: \(transforms.count)")
            print("shouldCenter: \(shouldCenterTrackedFeature)")
            if shouldCenterTrackedFeature && !compensationTrajectory.isEmpty {
                let step = max(1, compensationTrajectory.count / 20)
                for si in stride(from: 0, to: compensationTrajectory.count, by: step) {
                    let cx = compensationTrajectory[si].x, cy = compensationTrajectory[si].y
                    let dx = (centerX - cx) * Float(width), dy = (cy - centerY) * Float(height)
                    let tx = transforms[si].x, ty = transforms[si].y
                    print("帧\(si): 轨迹=(\(String(format:"%.4f",cx)),\(String(format:"%.4f",cy))) 期望dx=\(String(format:"%.1f",dx)) dy=\(String(format:"%.1f",dy)) 实际tx=\(String(format:"%.1f",tx)) ty=\(String(format:"%.1f",ty))")
                }
                let rawStep = max(1, rawTrajectory.count / 20)
                print("--- 原始轨迹(平滑前) ---")
                for si in stride(from: 0, to: rawTrajectory.count, by: rawStep) {
                    print("原始帧\(si): (\(String(format:"%.4f",rawTrajectory[si].x)),\(String(format:"%.4f",rawTrajectory[si].y)))")
                }
            }
            if !transforms.isEmpty {
                let maxDx = transforms.map { abs($0.x) }.max() ?? 0
                let maxDy = transforms.map { abs($0.y) }.max() ?? 0
                print("最大位移: (\(maxDx), \(maxDy))")
                print("首帧transform: (\(transforms[0].x), \(transforms[0].y))")
                let midI = transforms.count / 2
                print("中帧transform: (\(transforms[midI].x), \(transforms[midI].y))")
                let lastI = transforms.count - 1
                print("末帧transform: (\(transforms[lastI].x), \(transforms[lastI].y))")
            }
        }
        
        // ─── 第四步：渲染 ─────────────────────────────
        progressHandler(0.5, "GPU 渲染中...")
        
        // 计算裁切边距（基于绝对位移）
        let padX = robustPadding(for: transforms.map { $0.x }, extraMargin: 10)
        let padY = robustPadding(for: transforms.map { $0.y }, extraMargin: 10)
        let requiredPadRatio = max(padX / Float(width), padY / Float(height))
        let algorithmCropFloor: Float = shouldCenterTrackedFeature ? 0.04 : 0.015
        let preferredAutomaticCropRatio: Float = shouldCenterTrackedFeature ? 0.28 : 0.10
        let automaticCropTarget = max(requiredPadRatio, algorithmCropFloor)
        let safePadRatio = min(automaticCropTarget, preferredAutomaticCropRatio)
        let outputW = width
        let outputH = height
        let zoomDenominator = max(0.02, 1 - 2 * safePadRatio)
        let renderScale = CGFloat(1.0 / zoomDenominator)

        let maxTransformX = max(Float(width) * 0.46, 0)
        let maxTransformY = max(Float(height) * 0.46, 0)
        for i in 0..<transforms.count {
            transforms[i].x = max(-maxTransformX, min(maxTransformX, transforms[i].x))
            transforms[i].y = max(-maxTransformY, min(maxTransformY, transforms[i].y))
        }

        if shouldCenterTrackedFeature, desiredCenterTransforms.count == transforms.count {
            let residualDeadzone: Float = max(2.0, min(Float(max(width, height)) * 0.004, 6.0))
            let residualCorrectionLimit: Float = max(8.0, min(Float(max(width, height)) * 0.018, 18.0))
            let residualCorrectionGain: Float = 0.35
            for i in 0..<transforms.count {
                let residualX = desiredCenterTransforms[i].x - transforms[i].x
                let residualY = desiredCenterTransforms[i].y - transforms[i].y
                let limitedCorrectionX = residualCorrection(
                    for: residualX,
                    deadzone: residualDeadzone,
                    gain: residualCorrectionGain,
                    limit: residualCorrectionLimit
                )
                let limitedCorrectionY = residualCorrection(
                    for: residualY,
                    deadzone: residualDeadzone,
                    gain: residualCorrectionGain,
                    limit: residualCorrectionLimit
                )
                transforms[i].x = max(-maxTransformX, min(maxTransformX, transforms[i].x + limitedCorrectionX))
                transforms[i].y = max(-maxTransformY, min(maxTransformY, transforms[i].y + limitedCorrectionY))
            }
        }

        // 最终帧间增量限幅：防止遮挡恢复、跟踪突变等导致导出画面出现可见抖动尖峰。
        // 此处作为最后一道关卡，在残差修正完成后执行，不影响居中补偿的整体走势。
        let finalMaxDelta: Float = shouldCenterTrackedFeature
            ? max(Float(max(width, height)) * 0.005, 6.0)
            : max(Float(max(width, height)) * 0.003, 2.0)
        for i in 1..<transforms.count {
            let dx = transforms[i].x - transforms[i-1].x
            let dy = transforms[i].y - transforms[i-1].y
            let delta = hypot(dx, dy)
            if delta > finalMaxDelta {
                let scale = finalMaxDelta / delta
                transforms[i].x = transforms[i-1].x + dx * scale
                transforms[i].y = transforms[i-1].y + dy * scale
            }
        }

        let outputWidth = CGFloat(outputW)
        let outputHeight = CGFloat(outputH)
        let centerPixelX = outputWidth / 2
        let centerPixelY = outputHeight / 2
        let targetPoint = CGPoint(x: CGFloat(centerX), y: CGFloat(centerY))
        
        var centerErrors: [Double] = []
        let renderedTrajectory = filteredTrajectory.enumerated().map { index, point in
            let sourceX = CGFloat(point.x) * CGFloat(width) + CGFloat(transforms[index].x)
            let sourceY = (1 - CGFloat(point.y)) * CGFloat(height) + CGFloat(transforms[index].y)
            let renderedBottomY = ((sourceY - centerPixelY) * renderScale + centerPixelY) / outputHeight
            let rendered = CGPoint(
                x: ((sourceX - centerPixelX) * renderScale + centerPixelX) / outputWidth,
                y: 1 - renderedBottomY
            )
            let error = hypot(rendered.x - targetPoint.x, rendered.y - targetPoint.y)
            centerErrors.append(Double(error))
            return rendered
        }

        var transformDeltas: [Double] = []
        if transforms.count > 1 {
            for i in 1..<transforms.count {
                let dx = Double(transforms[i].x - transforms[i - 1].x)
                let dy = Double(transforms[i].y - transforms[i - 1].y)
                transformDeltas.append(hypot(dx, dy))
            }
        }

        lastAnalysis = StabilizationAnalysis(
            frameSize: CGSize(width: width, height: height),
            rawTrajectory: rawTrajectory.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) },
            smoothedTrajectory: smoothTrajectory.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) },
            renderedTrajectory: renderedTrajectory,
            transforms: transforms.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) },
            isCentering: shouldCenterTrackedFeature,
            cropRatio: Double(safePadRatio),
            cropLimited: automaticCropTarget > safePadRatio + 0.0001,
            meanCenterError: centerErrors.isEmpty ? 0 : centerErrors.reduce(0, +) / Double(centerErrors.count),
            maxCenterError: centerErrors.max() ?? 0,
            meanTransformDelta: transformDeltas.isEmpty ? 0 : transformDeltas.reduce(0, +) / Double(transformDeltas.count),
            maxTransformDelta: transformDeltas.max() ?? 0
        )
        
        guard outputW > 64 && outputH > 64 else {
            progressHandler(1.0, "错误：运动过大")
            return false
        }
        
        let writerFileType: AVFileType
        let videoCodec: AVVideoCodecType
        switch exportFormat.lowercased() {
        case "mov":
            writerFileType = .mov
            videoCodec = .h264
        case "prores":
            writerFileType = .mov
            videoCodec = .proRes422HQ
        default:
            writerFileType = .mp4
            videoCodec = .h264
        }
        
        try? FileManager.default.removeItem(at: outputURL)
        
        guard let writer = try? AVAssetWriter(outputURL: outputURL, fileType: writerFileType) else {
            progressHandler(1.0, "错误：无法创建输出")
            return false
        }
        
        let videoSettings: [String: Any] = [
            AVVideoCodecKey: videoCodec,
            AVVideoWidthKey: outputW,
            AVVideoHeightKey: outputH,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: Int(Double(outputW * outputH) * fps * 0.1 * bitrateMultiplier),
                AVVideoExpectedSourceFrameRateKey: Int(fps),
                AVVideoMaxKeyFrameIntervalKey: Int(fps)
            ]
        ]
        
        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        writerInput.expectsMediaDataInRealTime = false
        writerInput.transform = preferredTransform
        
        let adaptorAttrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: outputW,
            kCVPixelBufferHeightKey as String: outputH,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: adaptorAttrs
        )
        writer.add(writerInput)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
        
        guard let reader2 = try? AVAssetReader(asset: asset) else {
            progressHandler(1.0, "错误：无法重新读取")
            return false
        }
        
        let out2 = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: settings)
        reader2.add(out2)
        reader2.timeRange = clipTimeRange
        reader2.startReading()
        
        var frameIndex = 0
        let totalFrames = transforms.count
        let renderProgressInterval = max(1, totalFrames / 120)

        // 创建像素缓冲池，避免每帧分配内存
        var renderPool: CVPixelBufferPool?
        let poolAttrs: [String: Any] = [
            kCVPixelBufferPoolMinimumBufferCountKey as String: 3
        ]
        let bufAttrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: outputW,
            kCVPixelBufferHeightKey as String: outputH,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]
        CVPixelBufferPoolCreate(
            kCFAllocatorDefault,
            poolAttrs as CFDictionary,
            bufAttrs as CFDictionary,
            &renderPool
        )

        while let sample = out2.copyNextSampleBuffer() {
            autoreleasepool {
                let pts = CMSampleBufferGetPresentationTimeStamp(sample)

                let tx: Float
                let ty: Float
                if frameIndex < totalFrames {
                    tx = transforms[frameIndex].x
                    ty = transforms[frameIndex].y
                } else {
                    tx = 0
                    ty = 0
                }

                if let inBuf = CMSampleBufferGetImageBuffer(sample),
                   let outBuf = applyGPULegacy(
                        input: inBuf,
                        width: width, height: height,
                        outputW: outputW, outputH: outputH,
                       shiftX: CGFloat(tx), shiftY: CGFloat(ty),
                       scale: renderScale,
                       pixelBufferPool: renderPool
                   ) {
                    let presentationTime = adjustedPresentationTime(for: pts, relativeTo: clipStartTime)
                    while !adaptor.assetWriterInput.isReadyForMoreMediaData {
                        Thread.sleep(forTimeInterval: 0.01)
                    }
                    adaptor.append(outBuf, withPresentationTime: presentationTime)
                }

                frameIndex += 1

                if frameIndex == 1 || frameIndex == totalFrames || frameIndex % renderProgressInterval == 0 {
                    let prog = 0.5 + Double(frameIndex) / Double(totalFrames) * 0.5
                    progressHandler(min(prog, 1.0), "渲染中 \(frameIndex)/\(totalFrames)")
                }
            }
        }
        
        writerInput.markAsFinished()
        let sem = DispatchSemaphore(value: 0)
        writer.finishWriting { sem.signal() }
        sem.wait()
        
        if FileManager.default.fileExists(atPath: outputURL.path) {
            if videoInfo.audioTrack != nil {
                progressHandler(0.98, "合并原始音频...")
                let merged = muxOriginalAudioIfPossible(
                    from: asset,
                    into: outputURL,
                    clipTimeRange: clipTimeRange,
                    outputFileType: writerFileType
                )
                if debugMode && !merged {
                    print("警告: 原始音频合并失败，保留静音视频输出")
                }
            }
            progressHandler(1.0, "完成")
            return true
        } else {
            progressHandler(1.0, "错误：写入失败")
            return false
        }
    }
    
    // GPU 轨迹平滑
    private func smoothGPU(_ input: [(x: Float, y: Float)], windowSize: Int) -> [(x: Float, y: Float)] {
        guard !input.isEmpty else { return input }
        
        var result = input
        
        // 移动平均
        for i in 0..<input.count {
            var sumX: Float = 0
            var sumY: Float = 0
            var count: Float = 0
            
            for j in max(0, i - windowSize)...min(input.count - 1, i + windowSize) {
                sumX += input[j].x
                sumY += input[j].y
                count += 1
            }
            
            result[i].x = sumX / count
            result[i].y = sumY / count
        }
        
        // 指数平滑
        var smooth = result
        let alpha: Float = 0.1
        
        for i in 1..<result.count {
            smooth[i].x = smooth[i-1].x * (1 - alpha) + result[i].x * alpha
            smooth[i].y = smooth[i-1].y * (1 - alpha) + result[i].y * alpha
        }
        
        return smooth
    }

    private func repairFrozenTrajectorySegments(
        _ input: [(x: Float, y: Float)]
    ) -> [(x: Float, y: Float)] {
        guard input.count > 12 else { return input }

        var result = input
        let freezeStepThreshold: Float = 0.00018
        let minimumFrozenLength = 6
        let postJumpThreshold: Float = 0.0032
        // 长冻结阈值：超过此长度的冻结段即使没有跳变也需要修复
        let longFreezeThreshold = 20

        var i = 1
        while i < result.count - 2 {
            let prev = result[i - 1]
            let curr = result[i]
            if hypot(curr.x - prev.x, curr.y - prev.y) > freezeStepThreshold {
                i += 1
                continue
            }

            let runStart = i - 1
            var runEnd = i
            while runEnd + 1 < result.count {
                let a = result[runEnd]
                let b = result[runEnd + 1]
                if hypot(b.x - a.x, b.y - a.y) > freezeStepThreshold {
                    break
                }
                runEnd += 1
            }

            let frozenLength = runEnd - runStart

            // ── Case 1: 冻结后跳变 (原有逻辑) ──
            let hasContext = runStart >= 2 && runEnd + 2 < result.count
            if frozenLength >= minimumFrozenLength && hasContext {
                let before = result[runStart - 1]
                let atStart = result[runStart]
                let after = result[runEnd + 1]
                let afterNext = result[runEnd + 2]
                let jumpAfterFreeze = hypot(after.x - atStart.x, after.y - atStart.y)

                if jumpAfterFreeze > postJumpThreshold {
                    let startVelocity = (
                        x: atStart.x - before.x,
                        y: atStart.y - before.y
                    )
                    let endVelocity = (
                        x: afterNext.x - after.x,
                        y: afterNext.y - after.y
                    )
                    let span = Float((runEnd + 1) - runStart + 1)

                    for j in runStart...runEnd {
                        let t = Float(j - runStart + 1) / span
                        let t2 = t * t
                        let t3 = t2 * t
                        let h00 = 2 * t3 - 3 * t2 + 1
                        let h10 = t3 - 2 * t2 + t
                        let h01 = -2 * t3 + 3 * t2
                        let h11 = t3 - t2

                        result[j].x =
                            h00 * atStart.x +
                            h10 * startVelocity.x * span +
                            h01 * after.x +
                            h11 * endVelocity.x * span
                        result[j].y =
                            h00 * atStart.y +
                            h10 * startVelocity.y * span +
                            h01 * after.y +
                            h11 * endVelocity.y * span
                    }
                    i = runEnd + 1
                    continue
                }
            }

            // ── Case 2: 长时间冻结但没有明显跳变 (0107 模式) ──
            // 找到冻结段前后最近的"活跃"（非冻结）锚点，线性插值填充
            if frozenLength >= longFreezeThreshold {
                // 向前找最近的活跃锚点
                var anchorBefore: (index: Int, point: (x: Float, y: Float))?
                if runStart > 0 {
                    // 继续向前跳过更早的冻结段
                    var ab = runStart - 1
                    while ab > 0 {
                        let d = hypot(result[ab].x - result[ab - 1].x, result[ab].y - result[ab - 1].y)
                        if d > freezeStepThreshold { break }
                        ab -= 1
                    }
                    // 确认锚点不是同一冻结值
                    if ab >= 0 && hypot(result[ab].x - result[runStart].x, result[ab].y - result[runStart].y) > freezeStepThreshold * 3 {
                        anchorBefore = (ab, result[ab])
                    }
                }

                // 向后找最近的活跃锚点
                var anchorAfter: (index: Int, point: (x: Float, y: Float))?
                if runEnd + 1 < result.count {
                    var aa = runEnd + 1
                    // 跳过紧接着的另一段冻结
                    while aa + 1 < result.count {
                        let d = hypot(result[aa + 1].x - result[aa].x, result[aa + 1].y - result[aa].y)
                        if d > freezeStepThreshold { break }
                        aa += 1
                    }
                    if aa < result.count && hypot(result[aa].x - result[runStart].x, result[aa].y - result[runStart].y) > freezeStepThreshold * 3 {
                        anchorAfter = (aa, result[aa])
                    }
                }

                if let before = anchorBefore, let after = anchorAfter {
                    // 有前后锚点 → 线性插值
                    let totalSpan = Float(after.index - before.index)
                    if totalSpan > 0 {
                        for j in runStart...runEnd {
                            let t = Float(j - before.index) / totalSpan
                            result[j].x = before.point.x + (after.point.x - before.point.x) * t
                            result[j].y = before.point.y + (after.point.y - before.point.y) * t
                        }
                        if debugMode {
                            print("repairFrozen: linear interpolated long freeze [\(runStart)...\(runEnd)] between anchors \(before.index) and \(after.index)")
                        }
                    }
                } else if let after = anchorAfter {
                    // 只有后锚点（冻结从开头开始）→ 从冻结值渐变到后锚点
                    let frozen = result[runStart]
                    let totalSpan = Float(after.index - runStart)
                    if totalSpan > 0 {
                        for j in runStart...runEnd {
                            let t = Float(j - runStart) / totalSpan
                            // 用 ease-in 曲线，越靠近冻结段开头越保守
                            let easeT = t * t
                            result[j].x = frozen.x + (after.point.x - frozen.x) * easeT
                            result[j].y = frozen.y + (after.point.y - frozen.y) * easeT
                        }
                        if debugMode {
                            print("repairFrozen: ease-in from frozen start [\(runStart)...\(runEnd)] toward anchor \(after.index)")
                        }
                    }
                } else if let before = anchorBefore {
                    // 只有前锚点（冻结到结尾）→ 从前锚点渐变
                    let frozen = result[runStart]
                    let totalSpan = Float(runEnd - before.index)
                    if totalSpan > 0 {
                        for j in runStart...runEnd {
                            let t = Float(j - before.index) / totalSpan
                            // ease-out: 越远离锚点越保守
                            let easeT = 1 - (1 - t) * (1 - t)
                            result[j].x = before.point.x + (frozen.x - before.point.x) * easeT
                            result[j].y = before.point.y + (frozen.y - before.point.y) * easeT
                        }
                        if debugMode {
                            print("repairFrozen: ease-out from anchor \(before.index) through freeze [\(runStart)...\(runEnd)]")
                        }
                    }
                }
                // 如果前后都没有锚点，保持不变（整段视频都冻结，无法修复）
            }

            i = runEnd + 1
        }

        return result
    }

    private func suppressTrajectoryOutliers(
        _ input: [(x: Float, y: Float)]
    ) -> [(x: Float, y: Float)] {
        guard input.count > 4 else { return input }

        var result = input

        for index in 1..<(input.count - 1) {
            let previous = result[index - 1]
            let current = input[index]
            let next = input[index + 1]

            let expectedX = (previous.x + next.x) * 0.5
            let expectedY = (previous.y + next.y) * 0.5
            let expected = (x: expectedX, y: expectedY)

            let deviation = hypot(current.x - expected.x, current.y - expected.y)
            let neighborSpan = hypot(next.x - previous.x, next.y - previous.y)
            let prevJump = hypot(current.x - previous.x, current.y - previous.y)
            let nextJump = hypot(next.x - current.x, next.y - current.y)

            let incomingX = current.x - previous.x
            let incomingY = current.y - previous.y
            let outgoingX = next.x - current.x
            let outgoingY = next.y - current.y
            let directionFlip = (incomingX * outgoingX + incomingY * outgoingY) < 0

            let threshold = max(0.010, neighborSpan * 1.8 + 0.004)
            let isIsolatedSpike = deviation > threshold
                && prevJump > threshold * 0.85
                && nextJump > threshold * 0.85

            if isIsolatedSpike && (directionFlip || deviation > threshold * 1.5) {
                result[index] = expected
            }
        }

        return result
    }

    private func refineCenterLockTrajectory(
        _ input: [(x: Float, y: Float)],
        fps: Double,
        frameWidth: Int,
        frameHeight: Int
    ) -> [(x: Float, y: Float)] {
        guard input.count > 6 else { return input }

        let maxDimension = Float(max(frameWidth, frameHeight))
        let spikeThresholdPx = max(4.0, min(9.0, maxDimension * 0.0018))
        let medianRadius = max(2, min(5, Int((fps * 0.05).rounded(.toNearestOrEven))))
        var despiked = input

        for index in 0..<input.count {
            let lower = max(0, index - medianRadius)
            let upper = min(input.count - 1, index + medianRadius)
            let medianX = median(input[lower...upper].map { $0.x })
            let medianY = median(input[lower...upper].map { $0.y })
            let deviationPx = normalizedPixelDistance(
                input[index],
                (x: medianX, y: medianY),
                frameWidth: frameWidth,
                frameHeight: frameHeight
            )

            if deviationPx > spikeThresholdPx {
                let excess = (deviationPx - spikeThresholdPx) / max(spikeThresholdPx, 1)
                let weight = min(0.85, 0.35 + excess * 0.35)
                despiked[index].x = input[index].x * (1 - weight) + medianX * weight
                despiked[index].y = input[index].y * (1 - weight) + medianY * weight
            }
        }

        let forward = centerLockFilterPass(
            despiked,
            frameWidth: frameWidth,
            frameHeight: frameHeight
        )
        let backward = Array(centerLockFilterPass(
            Array(despiked.reversed()),
            frameWidth: frameWidth,
            frameHeight: frameHeight
        ).reversed())

        var result = input
        for index in 0..<result.count {
            result[index].x = (forward[index].x + backward[index].x) * 0.5
            result[index].y = (forward[index].y + backward[index].y) * 0.5
        }

        let startupFrames = min(result.count, max(0, Int((fps * 0.75).rounded(.toNearestOrEven))))
        if startupFrames > 6 {
            var earlyDeltas: [Float] = []
            for index in 1..<startupFrames {
                earlyDeltas.append(normalizedPixelDistance(
                    result[index],
                    result[index - 1],
                    frameWidth: frameWidth,
                    frameHeight: frameHeight
                ))
            }

            let sortedDeltas = earlyDeltas.sorted()
            let p90Index = min(sortedDeltas.count - 1, max(0, Int(Double(sortedDeltas.count - 1) * 0.90)))
            let earlyP90 = sortedDeltas.isEmpty ? 0 : sortedDeltas[p90Index]
            if earlyP90 > spikeThresholdPx * 0.9 {
                let startupRadius = max(4, min(12, Int((fps * 0.10).rounded(.toNearestOrEven))))
                let startupSmooth = smoothTransformSeries(result, windowRadius: startupRadius)
                for index in 0..<startupFrames {
                    let t = Float(index) / Float(max(startupFrames - 1, 1))
                    let weight = 0.35 * (1 - t) * (1 - t)
                    result[index].x = result[index].x * (1 - weight) + startupSmooth[index].x * weight
                    result[index].y = result[index].y * (1 - weight) + startupSmooth[index].y * weight
                }
            }
        }

        return result
    }

    private func centerLockFilterPass(
        _ input: [(x: Float, y: Float)],
        frameWidth: Int,
        frameHeight: Int
    ) -> [(x: Float, y: Float)] {
        guard input.count > 2 else { return input }

        let maxDimension = Float(max(frameWidth, frameHeight))
        let softResidualPx = max(3.0, min(8.0, maxDimension * 0.0016))
        let hardResidualPx = max(8.0, min(20.0, maxDimension * 0.0040))
        var result = input
        var velocity = (
            x: input[1].x - input[0].x,
            y: input[1].y - input[0].y
        )

        for index in 1..<input.count {
            let predicted = (
                x: result[index - 1].x + velocity.x,
                y: result[index - 1].y + velocity.y
            )
            let residual = (
                x: input[index].x - predicted.x,
                y: input[index].y - predicted.y
            )
            let residualPx = normalizedPixelDistance(
                input[index],
                predicted,
                frameWidth: frameWidth,
                frameHeight: frameHeight
            )

            let gain: Float
            let velocityAlpha: Float
            if residualPx <= softResidualPx {
                gain = 0.82
                velocityAlpha = 0.32
            } else if residualPx <= hardResidualPx {
                gain = 0.50
                velocityAlpha = 0.18
            } else {
                gain = 0.25
                velocityAlpha = 0.10
            }

            result[index] = clampNormalizedPoint((
                x: predicted.x + residual.x * gain,
                y: predicted.y + residual.y * gain
            ))

            let measuredVelocity = (
                x: result[index].x - result[index - 1].x,
                y: result[index].y - result[index - 1].y
            )
            velocity.x = velocity.x * (1 - velocityAlpha) + measuredVelocity.x * velocityAlpha
            velocity.y = velocity.y * (1 - velocityAlpha) + measuredVelocity.y * velocityAlpha
        }

        return result
    }

    private func normalizedPixelDistance(
        _ lhs: (x: Float, y: Float),
        _ rhs: (x: Float, y: Float),
        frameWidth: Int,
        frameHeight: Int
    ) -> Float {
        let dx = (lhs.x - rhs.x) * Float(frameWidth)
        let dy = (lhs.y - rhs.y) * Float(frameHeight)
        return sqrt(dx * dx + dy * dy)
    }

    private func clampNormalizedPoint(
        _ point: (x: Float, y: Float)
    ) -> (x: Float, y: Float) {
        (
            x: max(0, min(1, point.x)),
            y: max(0, min(1, point.y))
        )
    }

    private func median(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count % 2 == 0 {
            return (sorted[middle - 1] + sorted[middle]) * 0.5
        }
        return sorted[middle]
    }

    private func smoothTransformSeries(
        _ input: [(x: Float, y: Float)],
        windowRadius: Int
    ) -> [(x: Float, y: Float)] {
        guard !input.isEmpty, windowRadius > 0 else { return input }

        var result = input
        for i in 0..<input.count {
            let lower = max(0, i - windowRadius)
            let upper = min(input.count - 1, i + windowRadius)
            var sumX: Float = 0
            var sumY: Float = 0
            var count: Float = 0

            for j in lower...upper {
                sumX += input[j].x
                sumY += input[j].y
                count += 1
            }

            result[i].x = sumX / count
            result[i].y = sumY / count
        }

        return result
    }

    private func residualCorrection(
        for residual: Float,
        deadzone: Float,
        gain: Float,
        limit: Float
    ) -> Float {
        let magnitude = abs(residual)
        guard magnitude > deadzone else {
            return 0
        }

        let signedResidual = residual > 0 ? 1.0 as Float : -1.0 as Float
        let effectiveMagnitude = (magnitude - deadzone) * gain
        return signedResidual * min(limit, effectiveMagnitude)
    }

    private func robustPadding(
        for values: [Float],
        extraMargin: Float
    ) -> Float {
        guard !values.isEmpty else {
            return extraMargin
        }

        let absoluteValues = values.map { abs($0) }.sorted()
        let percentileIndex = min(
            absoluteValues.count - 1,
            max(0, Int((Double(absoluteValues.count - 1) * 0.96).rounded(.toNearestOrEven)))
        )
        let percentileValue = absoluteValues[percentileIndex]
        let maxValue = absoluteValues.last ?? percentileValue
        let blendedValue = percentileValue * 0.85 + maxValue * 0.15
        return blendedValue + extraMargin
    }

    private func renderedPoint(
        for point: (x: Float, y: Float),
        transform: (x: Float, y: Float),
        frameWidth: Int,
        frameHeight: Int,
        outputWidth: CGFloat,
        outputHeight: CGFloat,
        renderScale: CGFloat,
        centerPixelX: CGFloat,
        centerPixelY: CGFloat
    ) -> CGPoint {
        let sourceX = CGFloat(point.x) * CGFloat(frameWidth) + CGFloat(transform.x)
        let sourceY = (1 - CGFloat(point.y)) * CGFloat(frameHeight) + CGFloat(transform.y)
        let renderedBottomY = ((sourceY - centerPixelY) * renderScale + centerPixelY) / outputHeight
        return CGPoint(
            x: ((sourceX - centerPixelX) * renderScale + centerPixelX) / outputWidth,
            y: 1 - renderedBottomY
        )
    }

    private struct GrayscaleFrame {
        let width: Int
        let height: Int
        let pixels: [UInt8]
    }

    private struct RectangleTemplateTracker {
        private enum MatchMode {
            case combined
            case referenceBiased
        }

        private struct MatchCandidate {
            let point: CGPoint
            let score: Double
            let scale: CGFloat
            let referenceScore: Double
            let referenceCorrelationScore: Double
        }

        private struct PatchSamplingKey: Hashable {
            let sourceWidth: Int
            let sourceHeight: Int
        }

        private struct PatchSamplingPlan {
            let columnOffsets: [Int]
            let rowOffsets: [Int]
        }

        struct TrackResult {
            let point: CGPoint
            let score: Double
            let lostFrames: Int
            let scale: CGFloat
        }

        let frameWidth: Int
        let frameHeight: Int
        let templateWidth: Int
        let templateHeight: Int
        let halfTemplateWidth: Int
        let halfTemplateHeight: Int
        let referenceTemplate: [UInt8]
        var adaptiveTemplate: [UInt8]
        var lastCenter: CGPoint
        var velocity: CGPoint = .zero
        var lastScore: Double = .greatestFiniteMagnitude
        var lostFrames: Int = 0
        var currentScale: CGFloat = 1.0
        var trackingFrameIndex: Int = 0
        var referenceDriftFrames: Int = 0
        private var patchSamplingPlans: [PatchSamplingKey: PatchSamplingPlan] = [:]

        init?(
            frame: GrayscaleFrame,
            initialCenter: CGPoint,
            boxSize: CGSize
        ) {
            frameWidth = frame.width
            frameHeight = frame.height
            templateWidth = max(12, Int(boxSize.width.rounded(.toNearestOrEven)))
            templateHeight = max(12, Int(boxSize.height.rounded(.toNearestOrEven)))
            halfTemplateWidth = templateWidth / 2
            halfTemplateHeight = templateHeight / 2
            lastCenter = initialCenter

            guard let initialPatch = RectangleTemplateTracker.extractPatch(
                from: frame,
                centerX: Int(initialCenter.x.rounded(.toNearestOrEven)),
                centerY: Int(initialCenter.y.rounded(.toNearestOrEven)),
                sourceWidth: templateWidth,
                sourceHeight: templateHeight,
                outputWidth: templateWidth,
                outputHeight: templateHeight
            ) else {
                return nil
            }

            referenceTemplate = initialPatch
            adaptiveTemplate = initialPatch
        }

        mutating func track(frame: GrayscaleFrame) -> TrackResult {
            trackingFrameIndex += 1
            let predictedX = lastCenter.x + velocity.x
            let predictedY = lastCenter.y + velocity.y
            let velocityMagnitude = hypot(velocity.x, velocity.y)
            let scaledTemplateWidth = max(12, Int((CGFloat(templateWidth) * currentScale).rounded(.toNearestOrEven)))
            let scaledTemplateHeight = max(12, Int((CGFloat(templateHeight) * currentScale).rounded(.toNearestOrEven)))
            let baseRadius = max(scaledTemplateWidth, scaledTemplateHeight) / 3
            let searchRadius = min(
                min(max(frame.width, frame.height) / 8, 60),
                max(baseRadius + Int(velocityMagnitude.rounded(.up)) + lostFrames * 5, 10)
            )
            let coarseStep = searchRadius > 20 ? 3 : 2
            let scaleCandidates = candidateScales()

            guard let coarseCenter = bestCenter(
                in: frame,
                around: CGPoint(x: predictedX, y: predictedY),
                radius: searchRadius,
                step: coarseStep,
                scales: [currentScale]
            ) else {
                lostFrames += 1
                velocity.x *= 0.5
                velocity.y *= 0.5
                let fallbackPoint = clampCenter(CGPoint(x: predictedX, y: predictedY))
                return TrackResult(
                    point: fallbackPoint,
                    score: Double.greatestFiniteMagnitude,
                    lostFrames: lostFrames,
                    scale: currentScale
                )
            }

            let refineRadius = max(2, coarseStep + 1)
            let refinedCenter = bestCenter(
                in: frame,
                around: coarseCenter.point,
                radius: refineRadius,
                step: 1,
                scales: scaleCandidates,
                mode: .combined
            ) ?? coarseCenter

            let predictedCenter = clampCenter(CGPoint(x: predictedX, y: predictedY))
            let referenceDriftDetected = refinedCenter.referenceScore > 26
            if referenceDriftDetected {
                referenceDriftFrames = min(referenceDriftFrames + 1, 6)
            } else {
                referenceDriftFrames = max(referenceDriftFrames - 1, 0)
            }

            let shouldReferenceReanchor = lostFrames == 0 && (
                trackingFrameIndex % 15 == 0 ||
                referenceDriftFrames >= 2
            )
            let referenceCenter: MatchCandidate?
            if shouldReferenceReanchor {
                let reanchorRadius = min(searchRadius + 8, 50)
                referenceCenter = bestCenter(
                    in: frame,
                    around: predictedCenter,
                    radius: reanchorRadius,
                    step: max(2, reanchorRadius / 10),
                    scales: scaleCandidates,
                    mode: .referenceBiased
                )
            } else {
                referenceCenter = nil
            }

            let resolvedCenter: MatchCandidate
            let usedReferenceAnchor: Bool
            if let referenceCenter {
                let referenceDeviation = hypot(
                    referenceCenter.point.x - predictedCenter.x,
                    referenceCenter.point.y - predictedCenter.y
                )
                let maxReferenceDeviation = CGFloat(max(templateWidth, templateHeight)) * (referenceDriftFrames >= 3 ? 1.65 : 1.35)
                let referenceClearlyBetter =
                    referenceCenter.referenceScore + 4 < refinedCenter.referenceScore ||
                    referenceCenter.score + 2 < refinedCenter.score
                if referenceDeviation <= maxReferenceDeviation && referenceClearlyBetter {
                    resolvedCenter = referenceCenter
                    usedReferenceAnchor = true
                } else {
                    resolvedCenter = refinedCenter
                    usedReferenceAnchor = false
                }
            } else {
                resolvedCenter = refinedCenter
                usedReferenceAnchor = false
            }

            let previousCenter = lastCenter
            let candidateDeviation = hypot(resolvedCenter.point.x - predictedCenter.x, resolvedCenter.point.y - predictedCenter.y)
            // lostFrames 越多，允许的偏移越大（目标可能旋转/缩放后位移较大）
            let baseTrustedDeviation = CGFloat(max(templateWidth, templateHeight)) * (lostFrames > 0 ? 2.2 : 1.4)
            let driftDeviationScale: CGFloat
            switch referenceDriftFrames {
            case 0...1:
                driftDeviationScale = 1.0
            case 2...3:
                driftDeviationScale = 0.90  // 旋转/外观漂移时适当放宽（原 0.82）
            default:
                driftDeviationScale = 0.78  // 原 0.68，避免旋转目标被误判为丢失
            }
            let maxTrustedDeviation = baseTrustedDeviation * driftDeviationScale
            let scoreImprovementLimit: Double
            switch referenceDriftFrames {
            case 0...1:
                scoreImprovementLimit = 1.35
            case 2...3:
                scoreImprovementLimit = 1.28
            default:
                scoreImprovementLimit = 1.22  // 外观变化时放宽相对分数要求
            }
            let scoreImprovedEnough = lastScore.isFinite == false || resolvedCenter.score <= lastScore * scoreImprovementLimit
            // 外观变化场景（旋转/缩放）：只要 candidate 比预测点的 score 更低且偏移合理，就接受
            let candidateBetterThanFreeze: Bool
            if let frozenScore = RectangleTemplateTracker.extractPatch(
                from: frame,
                centerX: Int(predictedCenter.x.rounded()),
                centerY: Int(predictedCenter.y.rounded()),
                sourceWidth: templateWidth, sourceHeight: templateHeight,
                outputWidth: templateWidth, outputHeight: templateHeight
            ).map({ patch -> Double in
                var err = 0
                for i in 0..<patch.count { err += abs(Int(patch[i]) - Int(adaptiveTemplate[i])) }
                return Double(err) / Double(max(patch.count, 1))
            }) {
                candidateBetterThanFreeze = resolvedCenter.score < frozenScore * 1.05
            } else {
                candidateBetterThanFreeze = true
            }
            let acceptCandidate =
                resolvedCenter.score < 32 ||
                (resolvedCenter.score < 48 && candidateDeviation <= maxTrustedDeviation) ||
                (scoreImprovedEnough && candidateDeviation <= maxTrustedDeviation * 0.60) ||
                (candidateBetterThanFreeze && candidateDeviation <= maxTrustedDeviation)
            let chosenCenter = acceptCandidate ? resolvedCenter.point : predictedCenter
            let chosenScale = acceptCandidate ? resolvedCenter.scale : currentScale

            velocity = CGPoint(
                x: chosenCenter.x - previousCenter.x,
                y: chosenCenter.y - previousCenter.y
            )
            lastCenter = chosenCenter
            lastScore = resolvedCenter.score
            currentScale = max(0.72, min(1.45, chosenScale))

            if usedReferenceAnchor {
                let reanchorAlpha = resolvedCenter.score < 24 ? 0.10 : 0.05
                refreshTemplate(from: frame, center: chosenCenter, scale: currentScale, alpha: reanchorAlpha)
                referenceDriftFrames = max(referenceDriftFrames - 3, 0)
            }

            if resolvedCenter.score < 24 {
                lostFrames = 0
                refreshTemplate(from: frame, center: chosenCenter, scale: currentScale, alpha: 0.08)
                referenceDriftFrames = max(referenceDriftFrames - 2, 0)
            } else if resolvedCenter.score < 44 {  // 原 36，旋转/缩放目标外观变化大，放宽丢失阈值
                lostFrames = max(0, lostFrames - 1)
            } else {
                lostFrames += 1
                velocity.x *= 0.6
                velocity.y *= 0.6
                referenceDriftFrames = min(referenceDriftFrames + 1, 6)
            }

            return TrackResult(
                point: chosenCenter,
                score: resolvedCenter.score,
                lostFrames: lostFrames,
                scale: currentScale
            )
        }

        mutating func applyExternalCorrection(
            center: CGPoint,
            frame: GrayscaleFrame,
            shouldRefreshTemplate: Bool
        ) {
            let clamped = clampCenter(center)
            velocity = CGPoint(
                x: clamped.x - lastCenter.x,
                y: clamped.y - lastCenter.y
            )
            let canRefreshDuringExternalCorrection = shouldRefreshTemplate && referenceDriftFrames <= 1
            lastCenter = clamped
            if canRefreshDuringExternalCorrection {
                refreshTemplate(from: frame, center: clamped, scale: currentScale, alpha: 0.05)
                lostFrames = max(0, lostFrames - 2)
                lastScore = min(lastScore, 28)
            }
        }

        private mutating func refreshTemplate(
            from frame: GrayscaleFrame,
            center: CGPoint,
            scale: CGFloat,
            alpha: Double
        ) {
            guard let patch = extractPatch(
                from: frame,
                centerX: Int(center.x.rounded(.toNearestOrEven)),
                centerY: Int(center.y.rounded(.toNearestOrEven)),
                sourceWidth: max(12, Int((CGFloat(templateWidth) * scale).rounded(.toNearestOrEven))),
                sourceHeight: max(12, Int((CGFloat(templateHeight) * scale).rounded(.toNearestOrEven)))
            ) else {
                return
            }

            let refreshWeight = referenceRefreshWeight(for: patch)
            guard refreshWeight > 0 else {
                return
            }
            let driftRefreshMultiplier = max(0.28, 1.0 - Double(referenceDriftFrames) * 0.16)
            let recoveryRefreshMultiplier = lostFrames > 0 ? 0.72 : 1.0
            let effectiveAlpha = alpha * refreshWeight * driftRefreshMultiplier * recoveryRefreshMultiplier

            for index in 0..<adaptiveTemplate.count {
                let blended = Double(adaptiveTemplate[index]) * (1 - effectiveAlpha) + Double(patch[index]) * effectiveAlpha
                adaptiveTemplate[index] = UInt8(max(0, min(255, Int(blended.rounded(.toNearestOrEven)))))
            }
        }

        private func referenceRefreshWeight(for patch: [UInt8]) -> Double {
            guard patch.count == referenceTemplate.count, !patch.isEmpty else {
                return 0
            }

            var referenceError = 0
            var patchSum = 0
            var referenceSum = 0
            for index in 0..<patch.count {
                patchSum += Int(patch[index])
                referenceSum += Int(referenceTemplate[index])
                referenceError += abs(Int(patch[index]) - Int(referenceTemplate[index]))
            }

            let pixelCount = max(patch.count, 1)
            let referenceScore = Double(referenceError) / Double(pixelCount)
            let patchMean = patchSum / pixelCount
            let referenceMean = referenceSum / pixelCount

            var cross: Double = 0
            var patchVariance: Double = 0
            var referenceVariance: Double = 0
            for index in 0..<patch.count {
                let patchCentered = Int(patch[index]) - patchMean
                let referenceCentered = Int(referenceTemplate[index]) - referenceMean
                cross += Double(patchCentered * referenceCentered)
                patchVariance += Double(patchCentered * patchCentered)
                referenceVariance += Double(referenceCentered * referenceCentered)
            }

            let referenceCorrelationScore = normalizedCorrelationScore(
                cross: cross,
                lhsVariance: patchVariance,
                rhsVariance: referenceVariance
            )

            if referenceScore < 20 && referenceCorrelationScore < 0.30 {
                return 1.0
            }
            if referenceScore < 28 && referenceCorrelationScore < 0.42 {
                return 0.45
            }
            if referenceScore < 36 && referenceCorrelationScore < 0.55 {
                return 0.18
            }

            return 0
        }

        private mutating func bestCenter(
            in frame: GrayscaleFrame,
            around center: CGPoint,
            radius: Int,
            step: Int,
            scales: [CGFloat],
            mode: MatchMode = .combined
        ) -> MatchCandidate? {
            let clampedCenter = clampCenter(center)
            let centerX = Int(clampedCenter.x.rounded(.toNearestOrEven))
            let centerY = Int(clampedCenter.y.rounded(.toNearestOrEven))
            let minX = max(halfTemplateWidth, centerX - radius)
            let maxX = min(frame.width - halfTemplateWidth - 1, centerX + radius)
            let minY = max(halfTemplateHeight, centerY - radius)
            let maxY = min(frame.height - halfTemplateHeight - 1, centerY + radius)

            guard minX <= maxX, minY <= maxY else {
                return nil
            }

            var bestPoint = clampedCenter
            var bestScore = Double.greatestFiniteMagnitude
            var bestScale = currentScale
            var bestReferenceScore = Double.greatestFiniteMagnitude
            var bestReferenceCorrelationScore = Double.greatestFiniteMagnitude
            var bestDistanceFromPrediction = Double.greatestFiniteMagnitude
            let searchStep = max(step, 1)
            let candidateXs = orderedSearchPositions(
                minValue: minX,
                maxValue: maxX,
                step: searchStep,
                anchor: centerX
            )
            let candidateYs = orderedSearchPositions(
                minValue: minY,
                maxValue: maxY,
                step: searchStep,
                anchor: centerY
            )

            for candidateY in candidateYs {
                for candidateX in candidateXs {
                    for scale in scales {
                        let tieWindow: Double = mode == .combined ? 0.45 : 0.30
                        guard let score = scoreCandidate(
                            in: frame,
                            centerX: candidateX,
                            centerY: candidateY,
                            scale: scale,
                            predictedCenter: clampedCenter,
                            searchRadius: radius,
                            currentBestScoreCutoff: bestScore.isFinite ? bestScore + tieWindow : Double.greatestFiniteMagnitude,
                            mode: mode
                        ) else {
                            continue
                        }
                        let candidateDistanceFromPrediction = Double(
                            hypot(
                                CGFloat(candidateX) - clampedCenter.x,
                                CGFloat(candidateY) - clampedCenter.y
                            )
                        )
                        let betterScore = score.score < bestScore - 0.0001
                        let nearTie = score.score <= bestScore + tieWindow
                        let referenceAdvantage: Bool
                        let referenceComparable: Bool
                        switch mode {
                        case .combined:
                            let driftMargin = referenceDriftFrames >= 2 ? 0.55 : 0.30
                            let comparableMargin = referenceDriftFrames >= 2 ? 0.90 : 0.45
                            referenceAdvantage = score.referenceScore + driftMargin < bestReferenceScore
                            referenceComparable = score.referenceScore <= bestReferenceScore + comparableMargin
                        case .referenceBiased:
                            referenceAdvantage = score.referenceScore + 0.25 < bestReferenceScore
                            referenceComparable = score.referenceScore <= bestReferenceScore + 0.25
                        }
                        let distanceAdvantage = candidateDistanceFromPrediction + 0.35 < bestDistanceFromPrediction
                        let betterTieBreak =
                            nearTie && (
                                referenceAdvantage ||
                                (referenceComparable && distanceAdvantage)
                            )

                        if betterScore || betterTieBreak {
                            bestScore = score.score
                            bestPoint = CGPoint(x: candidateX, y: candidateY)
                            bestScale = scale
                            bestReferenceScore = score.referenceScore
                            bestReferenceCorrelationScore = score.referenceCorrelationScore
                            bestDistanceFromPrediction = candidateDistanceFromPrediction
                        }
                    }
                }
            }

            guard bestScore.isFinite else {
                return nil
            }

            return MatchCandidate(
                point: bestPoint,
                score: bestScore,
                scale: bestScale,
                referenceScore: bestReferenceScore,
                referenceCorrelationScore: bestReferenceCorrelationScore
            )
        }

        private mutating func scoreCandidate(
            in frame: GrayscaleFrame,
            centerX: Int,
            centerY: Int,
            scale: CGFloat,
            predictedCenter: CGPoint,
            searchRadius: Int,
            currentBestScoreCutoff: Double,
            mode: MatchMode
        ) -> (score: Double, referenceScore: Double, referenceCorrelationScore: Double)? {
            let distanceFromPrediction = hypot(
                CGFloat(centerX) - predictedCenter.x,
                CGFloat(centerY) - predictedCenter.y
            )
            let normalizedDistance = Double(distanceFromPrediction) / Double(max(searchRadius, 1))
            let motionPenaltyWeight: Double
            switch lostFrames {
            case 0:
                motionPenaltyWeight = 2.8
            case 1:
                motionPenaltyWeight = 1.6
            default:
                motionPenaltyWeight = 0.6
            }
            let scalePenalty = Double(abs(scale - currentScale)) * (lostFrames == 0 ? 2.0 : 0.8)
            let geometricLowerBound = normalizedDistance * motionPenaltyWeight + scalePenalty
            let driftBias = min(Double(referenceDriftFrames) / 4.0, 1.0)
            let adaptiveScoreWeight: Double
            let referenceScoreWeight: Double
            switch mode {
            case .combined:
                adaptiveScoreWeight = 0.20 * (1 - driftBias) + 0.05 * driftBias
                referenceScoreWeight = 0.08 * (1 - driftBias) + 0.18 * driftBias
            case .referenceBiased:
                adaptiveScoreWeight = 0.05
                referenceScoreWeight = 0.18
            }
            if currentBestScoreCutoff.isFinite && geometricLowerBound > currentBestScoreCutoff {
                return nil
            }

            let sourceWidth = max(12, Int((CGFloat(templateWidth) * scale).rounded(.toNearestOrEven)))
            let sourceHeight = max(12, Int((CGFloat(templateHeight) * scale).rounded(.toNearestOrEven)))
            guard let patch = extractPatch(
                from: frame,
                centerX: centerX,
                centerY: centerY,
                sourceWidth: sourceWidth,
                sourceHeight: sourceHeight
            ) else {
                return nil
            }

            // Accelerate 向量化 SAD 计算
            let count = patch.count
            var adaptiveError = 0
            var referenceError = 0
            var patchSum = 0
            var adaptiveSum = 0
            var referenceSum = 0

            patch.withUnsafeBufferPointer { patchBuf in
                adaptiveTemplate.withUnsafeBufferPointer { adaptBuf in
                    referenceTemplate.withUnsafeBufferPointer { refBuf in
                        let pPtr = patchBuf.baseAddress!
                        let aPtr = adaptBuf.baseAddress!
                        let rPtr = refBuf.baseAddress!
                        for index in 0..<count {
                            let p = Int(pPtr[index])
                            let a = Int(aPtr[index])
                            let r = Int(rPtr[index])
                            patchSum &+= p
                            adaptiveSum &+= a
                            referenceSum &+= r
                            adaptiveError &+= abs(p &- a)
                            referenceError &+= abs(p &- r)
                        }
                    }
                }
            }

            let pixelCount = max(patch.count, 1)
            let adaptiveScore = Double(adaptiveError) / Double(pixelCount)
            let referenceScore = Double(referenceError) / Double(pixelCount)
            if currentBestScoreCutoff.isFinite {
                let lowerBound =
                    geometricLowerBound
                    + adaptiveScoreWeight * adaptiveScore
                    + referenceScoreWeight * referenceScore
                if lowerBound > currentBestScoreCutoff {
                    return nil
                }
            }
            let patchMean = patchSum / pixelCount
            let adaptiveMean = adaptiveSum / pixelCount
            let referenceMean = referenceSum / pixelCount

            var adaptiveTextureError = 0
            var referenceTextureError = 0
            var adaptiveCross: Double = 0
            var referenceCross: Double = 0
            var patchVariance: Double = 0
            var adaptiveVariance: Double = 0
            var referenceVariance: Double = 0
            for index in 0..<patch.count {
                let patchCentered = Int(patch[index]) - patchMean
                let adaptiveCentered = Int(adaptiveTemplate[index]) - adaptiveMean
                let referenceCentered = Int(referenceTemplate[index]) - referenceMean
                adaptiveTextureError += abs(patchCentered - adaptiveCentered)
                referenceTextureError += abs(patchCentered - referenceCentered)
                adaptiveCross += Double(patchCentered * adaptiveCentered)
                referenceCross += Double(patchCentered * referenceCentered)
                patchVariance += Double(patchCentered * patchCentered)
                adaptiveVariance += Double(adaptiveCentered * adaptiveCentered)
                referenceVariance += Double(referenceCentered * referenceCentered)
            }

            let adaptiveTextureScore = Double(adaptiveTextureError) / Double(pixelCount)
            let referenceTextureScore = Double(referenceTextureError) / Double(pixelCount)
            let adaptiveCorrelationScore = normalizedCorrelationScore(
                cross: adaptiveCross,
                lhsVariance: patchVariance,
                rhsVariance: adaptiveVariance
            )
            let referenceCorrelationScore = normalizedCorrelationScore(
                cross: referenceCross,
                lhsVariance: patchVariance,
                rhsVariance: referenceVariance
            )

            var adaptiveGradientError = 0
            var referenceGradientError = 0
            if templateWidth > 2 && templateHeight > 2 {
                for row in 1..<(templateHeight - 1) {
                    let rowOffset = row * templateWidth
                    for column in 1..<(templateWidth - 1) {
                        let index = rowOffset + column
                        let left = index - 1
                        let right = index + 1
                        let top = index - templateWidth
                        let bottom = index + templateWidth

                        let patchGradX = Int(patch[right]) - Int(patch[left])
                        let patchGradY = Int(patch[bottom]) - Int(patch[top])
                        let adaptiveGradX = Int(adaptiveTemplate[right]) - Int(adaptiveTemplate[left])
                        let adaptiveGradY = Int(adaptiveTemplate[bottom]) - Int(adaptiveTemplate[top])
                        let referenceGradX = Int(referenceTemplate[right]) - Int(referenceTemplate[left])
                        let referenceGradY = Int(referenceTemplate[bottom]) - Int(referenceTemplate[top])

                        adaptiveGradientError += abs(patchGradX - adaptiveGradX) + abs(patchGradY - adaptiveGradY)
                        referenceGradientError += abs(patchGradX - referenceGradX) + abs(patchGradY - referenceGradY)
                    }
                }
            }

            let gradientPixelCount = max((templateWidth - 2) * (templateHeight - 2) * 2, 1)
            let adaptiveGradientScore = Double(adaptiveGradientError) / Double(gradientPixelCount)
            let referenceGradientScore = Double(referenceGradientError) / Double(gradientPixelCount)

            let combinedAppearanceScore = adaptiveScore * 0.20
                + referenceScore * 0.08
                + adaptiveTextureScore * 0.22
                + referenceTextureScore * 0.10
                + adaptiveGradientScore * 0.22
                + referenceGradientScore * 0.10
                + adaptiveCorrelationScore * 18.0
                + referenceCorrelationScore * 10.0
            let referenceBiasedAppearanceScore = adaptiveScore * 0.05
                + referenceScore * 0.18
                + adaptiveTextureScore * 0.08
                + referenceTextureScore * 0.24
                + adaptiveGradientScore * 0.06
                + referenceGradientScore * 0.22
                + adaptiveCorrelationScore * 4.0
                + referenceCorrelationScore * 20.0

            let appearanceScore: Double
            switch mode {
            case .combined:
                appearanceScore = combinedAppearanceScore * (1 - driftBias)
                    + referenceBiasedAppearanceScore * driftBias
            case .referenceBiased:
                appearanceScore = referenceBiasedAppearanceScore
            }
            let totalScore = appearanceScore
                + geometricLowerBound

            return (totalScore, referenceScore, referenceCorrelationScore)
        }

        private func normalizedCorrelationScore(
            cross: Double,
            lhsVariance: Double,
            rhsVariance: Double
        ) -> Double {
            let epsilon = 1e-6
            let denominator = max(epsilon, sqrt(lhsVariance * rhsVariance))
            let correlation = max(-1.0, min(1.0, cross / denominator))
            return 1.0 - correlation
        }

        private func orderedSearchPositions(
            minValue: Int,
            maxValue: Int,
            step: Int,
            anchor: Int
        ) -> [Int] {
            let resolvedStep = max(step, 1)
            let count = max(0, (maxValue - minValue) / resolvedStep + 1)
            guard count > 0 else {
                return []
            }

            let clampedAnchor = min(max(minValue, anchor), maxValue)
            var positions: [Int] = []
            positions.reserveCapacity(count)

            var leftIndex = min(count - 1, max(0, (clampedAnchor - minValue) / resolvedStep))
            var rightIndex = leftIndex + 1

            while leftIndex >= 0 || rightIndex < count {
                let leftValue = leftIndex >= 0 ? (minValue + leftIndex * resolvedStep) : nil
                let rightValue = rightIndex < count ? (minValue + rightIndex * resolvedStep) : nil

                switch (leftValue, rightValue) {
                case let (lhs?, rhs?):
                    let lhsDistance = abs(lhs - clampedAnchor)
                    let rhsDistance = abs(rhs - clampedAnchor)
                    if lhsDistance <= rhsDistance {
                        positions.append(lhs)
                        leftIndex -= 1
                    } else {
                        positions.append(rhs)
                        rightIndex += 1
                    }
                case let (lhs?, nil):
                    positions.append(lhs)
                    leftIndex -= 1
                case let (nil, rhs?):
                    positions.append(rhs)
                    rightIndex += 1
                case (nil, nil):
                    break
                }
            }

            return positions
        }

        private mutating func patchSamplingPlan(
            sourceWidth: Int,
            sourceHeight: Int
        ) -> PatchSamplingPlan {
            let key = PatchSamplingKey(sourceWidth: sourceWidth, sourceHeight: sourceHeight)
            if let existing = patchSamplingPlans[key] {
                return existing
            }

            let widthScale = Double(sourceWidth) / Double(max(templateWidth, 1))
            let heightScale = Double(sourceHeight) / Double(max(templateHeight, 1))
            var columnOffsets = [Int](repeating: 0, count: templateWidth)
            var rowOffsets = [Int](repeating: 0, count: templateHeight)

            for column in 0..<templateWidth {
                columnOffsets[column] = min(
                    sourceWidth - 1,
                    max(0, Int((Double(column) + 0.5) * widthScale))
                )
            }
            for row in 0..<templateHeight {
                rowOffsets[row] = min(
                    sourceHeight - 1,
                    max(0, Int((Double(row) + 0.5) * heightScale))
                )
            }

            let plan = PatchSamplingPlan(
                columnOffsets: columnOffsets,
                rowOffsets: rowOffsets
            )
            patchSamplingPlans[key] = plan
            return plan
        }

        private mutating func extractPatch(
            from frame: GrayscaleFrame,
            centerX: Int,
            centerY: Int,
            sourceWidth: Int,
            sourceHeight: Int
        ) -> [UInt8]? {
            let halfWidth = sourceWidth / 2
            let halfHeight = sourceHeight / 2
            let minX = centerX - halfWidth
            let minY = centerY - halfHeight
            let maxX = minX + sourceWidth
            let maxY = minY + sourceHeight

            guard minX >= 0, minY >= 0, maxX <= frame.width, maxY <= frame.height else {
                return nil
            }

            let plan = patchSamplingPlan(sourceWidth: sourceWidth, sourceHeight: sourceHeight)
            var patch = [UInt8](repeating: 0, count: templateWidth * templateHeight)
            var outputIndex = 0

            for rowOffset in plan.rowOffsets {
                let rowBase = (minY + rowOffset) * frame.width + minX
                for columnOffset in plan.columnOffsets {
                    patch[outputIndex] = frame.pixels[rowBase + columnOffset]
                    outputIndex += 1
                }
            }

            return patch
        }

        private func candidateScales() -> [CGFloat] {
            let base = max(0.72, min(1.45, currentScale))
            let step: CGFloat = lostFrames > 0 ? 0.08 : 0.05
            let candidates = [
                max(0.72, base - step),
                base,
                min(1.45, base + step)
            ]
            var unique: [CGFloat] = []
            for value in candidates {
                if !unique.contains(where: { abs($0 - value) < 0.0001 }) {
                    unique.append(value)
                }
            }
            return unique
        }

        private func clampCenter(_ point: CGPoint) -> CGPoint {
            CGPoint(
                x: min(max(CGFloat(halfTemplateWidth), point.x), CGFloat(frameWidth - halfTemplateWidth - 1)),
                y: min(max(CGFloat(halfTemplateHeight), point.y), CGFloat(frameHeight - halfTemplateHeight - 1))
            )
        }

        private static func extractPatch(
            from frame: GrayscaleFrame,
            centerX: Int,
            centerY: Int,
            sourceWidth: Int,
            sourceHeight: Int,
            outputWidth: Int,
            outputHeight: Int
        ) -> [UInt8]? {
            let halfWidth = sourceWidth / 2
            let halfHeight = sourceHeight / 2
            let minX = centerX - halfWidth
            let minY = centerY - halfHeight
            let maxX = minX + sourceWidth
            let maxY = minY + sourceHeight

            guard minX >= 0, minY >= 0, maxX <= frame.width, maxY <= frame.height else {
                return nil
            }

            var patch = [UInt8]()
            patch.reserveCapacity(outputWidth * outputHeight)
            let widthScale = Double(sourceWidth) / Double(max(outputWidth, 1))
            let heightScale = Double(sourceHeight) / Double(max(outputHeight, 1))

            for row in 0..<outputHeight {
                let sampleY = min(
                    maxY - 1,
                    max(minY, minY + Int((Double(row) + 0.5) * heightScale))
                )
                let rowBase = sampleY * frame.width
                for column in 0..<outputWidth {
                    let sampleX = min(
                        maxX - 1,
                        max(minX, minX + Int((Double(column) + 0.5) * widthScale))
                    )
                    patch.append(frame.pixels[rowBase + sampleX])
                }
            }
            return patch
        }
    }

    private func trackTrajectoryWithTemplateMatching(
        asset: AVAsset,
        videoTrack: AVAssetTrack,
        settings: [String: Any],
        startTime: Double,
        endTime: Double,
        clipDuration: Double,
        fps: Double,
        initialPoint: CGPoint,
        trackingBoxSize: CGSize,
        frameSize: CGSize,
        progressHandler: @escaping (Double, String) -> Void
    ) -> [CGPoint]? {
        guard let reader = try? AVAssetReader(asset: asset) else {
            return nil
        }

        let output = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: settings)
        reader.add(output)
        reader.startReading()

        let downsampleStep = max(2, Int(max(frameSize.width, frameSize.height) / 480.0))
        let scaledBox = CGSize(
            width: max(16, trackingBoxSize.width / CGFloat(downsampleStep)),
            height: max(16, trackingBoxSize.height / CGFloat(downsampleStep))
        )
        let initialCenter = CGPoint(
            x: initialPoint.x / CGFloat(downsampleStep),
            y: initialPoint.y / CGFloat(downsampleStep)
        )

        var trajectory: [CGPoint] = []
        var tracker: RectangleTemplateTracker?
        let gpuTracker = GPUStabilizer()
        gpuTracker?.debugMode = debugMode
        gpuTracker?.debugMode = debugMode
        var previousTexture: MTLTexture?
        var lastNormalizedPoint = CGPoint(
            x: initialPoint.x / frameSize.width,
            y: initialPoint.y / frameSize.height
        )
        var frameCount = 0
        let totalFrames = Int(clipDuration * fps)
        let templateProgressInterval = max(1, totalFrames / 120)

        while let sample = output.copyNextSampleBuffer() {
            autoreleasepool {
                let pts = CMSampleBufferGetPresentationTimeStamp(sample)
                let ptsSec = CMTimeGetSeconds(pts)

                guard ptsSec >= startTime && ptsSec <= endTime else { return }
                guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample),
                      let grayFrame = makeGrayscaleFrame(from: pixelBuffer, step: downsampleStep) else { return }
                let currentTexture = gpuTracker?.createTexture(from: pixelBuffer)
                var gpuPoint: CGPoint?
                if let gpuTracker, let currentTexture {
                    if let previousTexture {
                        let tracked = gpuTracker.trackWithRecovery(
                            prev: previousTexture,
                            curr: currentTexture,
                            centerPoint: SIMD2<Float>(Float(lastNormalizedPoint.x), Float(lastNormalizedPoint.y))
                        )
                        gpuPoint = CGPoint(x: CGFloat(tracked.x), y: CGFloat(tracked.y))
                    } else {
                        gpuTracker.currentPoint = SIMD2<Float>(Float(lastNormalizedPoint.x), Float(lastNormalizedPoint.y))
                        gpuTracker.lastGoodPoint = gpuTracker.currentPoint
                        gpuPoint = lastNormalizedPoint
                    }
                }

                if tracker == nil {
                    tracker = RectangleTemplateTracker(
                        frame: grayFrame,
                        initialCenter: initialCenter,
                        boxSize: scaledBox
                    )
                }

                guard var activeTracker = tracker else { return }
                let templateResult = frameCount == 0
                    ? RectangleTemplateTracker.TrackResult(
                        point: activeTracker.lastCenter,
                        score: 0,
                        lostFrames: 0,
                        scale: activeTracker.currentScale
                    )
                    : activeTracker.track(frame: grayFrame)

                let templatePointNormalized = CGPoint(
                    x: templateResult.point.x / CGFloat(grayFrame.width),
                    y: templateResult.point.y / CGFloat(grayFrame.height)
                )

                let chosenPointNormalized: CGPoint
                if let gpuPoint {
                    let disagreement = hypot(
                        templatePointNormalized.x - gpuPoint.x,
                        templatePointNormalized.y - gpuPoint.y
                    )
                    let templateStrong = templateResult.lostFrames == 0 && templateResult.score < 26
                    let templateRecovering = templateResult.lostFrames <= 2 && templateResult.score < 34
                    let canTrustGPUCorrection = disagreement < 0.06 && templateResult.lostFrames >= 2

                    if templateStrong || (templateRecovering && disagreement < 0.04) {
                        let blend: CGFloat = templateStrong ? 0.86 : 0.65
                        chosenPointNormalized = CGPoint(
                            x: templatePointNormalized.x * blend + gpuPoint.x * (1 - blend),
                            y: templatePointNormalized.y * blend + gpuPoint.y * (1 - blend)
                        )
                    } else if canTrustGPUCorrection {
                        let correctedCenter = CGPoint(
                            x: gpuPoint.x * CGFloat(grayFrame.width),
                            y: gpuPoint.y * CGFloat(grayFrame.height)
                        )
                        activeTracker.applyExternalCorrection(
                            center: correctedCenter,
                            frame: grayFrame,
                            shouldRefreshTemplate: templateResult.lostFrames == 2 && disagreement < 0.04
                        )
                        chosenPointNormalized = gpuPoint
                    } else {
                        chosenPointNormalized = templatePointNormalized
                    }
                } else {
                    chosenPointNormalized = templatePointNormalized
                }

                tracker = activeTracker

                trajectory.append(chosenPointNormalized)
                lastNormalizedPoint = chosenPointNormalized
                previousTexture = currentTexture

                frameCount += 1
                if frameCount == 1 || frameCount == totalFrames || frameCount % templateProgressInterval == 0 {
                    let prog = 0.05 + Double(frameCount) / Double(max(totalFrames, 1)) * 0.4
                    progressHandler(min(prog, 0.45), "模板跟踪中 \(frameCount)/\(totalFrames)")
                }
            }
        }

        return trajectory.count > 2 ? trajectory : nil
    }

    private func makeGrayscaleFrame(
        from pixelBuffer: CVPixelBuffer,
        step: Int
    ) -> GrayscaleFrame? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return nil
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let scaledWidth = max(1, width / step)
        let scaledHeight = max(1, height / step)
        let source = baseAddress.assumingMemoryBound(to: UInt8.self)

        var pixels = [UInt8](repeating: 0, count: scaledWidth * scaledHeight)
        for y in 0..<scaledHeight {
            let sourceY = min(height - 1, y * step)
            let rowBase = source + sourceY * bytesPerRow
            for x in 0..<scaledWidth {
                let sourceX = min(width - 1, x * step)
                let pixelBase = rowBase + sourceX * 4
                let b = Int(pixelBase[0])
                let g = Int(pixelBase[1])
                let r = Int(pixelBase[2])
                pixels[y * scaledWidth + x] = UInt8((77 * r + 150 * g + 29 * b) >> 8)
            }
        }

        return GrayscaleFrame(width: scaledWidth, height: scaledHeight, pixels: pixels)
    }

    /// 从 preferredTransform 推导 Vision 所需的像素朝向。
    /// 原始像素数据按 naturalSize 排列，preferredTransform 描述播放器如何旋转显示。
    private func cgImageOrientation(from transform: CGAffineTransform) -> CGImagePropertyOrientation {
        let a = transform.a, b = transform.b, c = transform.c, d = transform.d
        // 90° CW 存储（手机竖拍常见）：b≈1, c≈-1
        if abs(b - 1) < 0.1 && abs(c + 1) < 0.1 { return .right }
        // 90° CCW 存储：b≈-1, c≈1
        if abs(b + 1) < 0.1 && abs(c - 1) < 0.1 { return .left }
        // 180° 存储：a≈-1, d≈-1
        if abs(a + 1) < 0.1 && abs(d + 1) < 0.1 { return .down }
        // 水平镜像：a≈-1, d≈1
        if abs(a + 1) < 0.1 && abs(d - 1) < 0.1 { return .upMirrored }
        // 垂直镜像：a≈1, d≈-1
        if abs(a - 1) < 0.1 && abs(d + 1) < 0.1 { return .downMirrored }
        return .up
    }

    private func trackTrajectoryWithVision(
        asset: AVAsset,
        videoTrack: AVAssetTrack,
        settings: [String: Any],
        startTime: Double,
        endTime: Double,
        clipDuration: Double,
        fps: Double,
        initialPoint: CGPoint,
        trackingBoxSize: CGSize,
        frameSize: CGSize,
        preferredTransform: CGAffineTransform = .identity,
        progressHandler: @escaping (Double, String) -> Void
    ) -> [CGPoint]? {
        guard let reader = try? AVAssetReader(asset: asset) else {
            return nil
        }
        
        let output = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: settings)
        reader.add(output)
        reader.startReading()
        
        // 小框保底：Vision 需要足够大的区域才能稳定跟踪（特别是旋转时）
        let minNormalizedDim: CGFloat = 0.06
        let normalizedBoxWidth = min(0.6, max(minNormalizedDim, trackingBoxSize.width / frameSize.width))
        let normalizedBoxHeight = min(0.6, max(minNormalizedDim, trackingBoxSize.height / frameSize.height))
        let normalizedCenterX = initialPoint.x / frameSize.width
        let normalizedCenterY = initialPoint.y / frameSize.height
        
        let initialRect = CGRect(
            x: max(0, min(1 - normalizedBoxWidth, normalizedCenterX - normalizedBoxWidth / 2)),
            y: max(0, min(1 - normalizedBoxHeight, 1 - normalizedCenterY - normalizedBoxHeight / 2)),
            width: normalizedBoxWidth,
            height: normalizedBoxHeight
        )
        
        let gpuTracker = GPUStabilizer()
        var request: VNTrackObjectRequest?
        let sequenceHandler = VNSequenceRequestHandler()
        var trajectory: [CGPoint] = []
        var lastPoint = CGPoint(x: normalizedCenterX, y: normalizedCenterY)
        var previousPoint = lastPoint
        var velocity = CGPoint.zero
        var lastBoundingBox = initialRect
        var lostFrames = 0
        var disagreementFrames = 0
        var visionFallbackFramesRemaining = 0
        var visionErrorStreak = 0
        var visionPermanentlyDisabled = false
        var frameCount = 0
        let totalFrames = Int(clipDuration * fps)
        let visionProgressInterval = max(1, totalFrames / 120)
        var previousTexture: MTLTexture?
        
        while let sample = output.copyNextSampleBuffer() {
            autoreleasepool {
                let pts = CMSampleBufferGetPresentationTimeStamp(sample)
                let ptsSec = CMTimeGetSeconds(pts)
                
                guard ptsSec >= startTime && ptsSec <= endTime else { return }
                guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { return }
                let currentTexture = gpuTracker?.createTexture(from: pixelBuffer)
                var gpuPoint: CGPoint?
                if let gpuTracker, let prevTexture = previousTexture, let currentTexture {
                    let tracked = gpuTracker.trackWithRecovery(
                        prev: prevTexture,
                        curr: currentTexture,
                        centerPoint: SIMD2<Float>(Float(lastPoint.x), Float(lastPoint.y))
                    )
                    gpuPoint = CGPoint(x: CGFloat(tracked.x), y: CGFloat(tracked.y))
                } else if let gpuTracker, currentTexture != nil {
                    gpuTracker.currentPoint = SIMD2<Float>(Float(lastPoint.x), Float(lastPoint.y))
                    gpuTracker.lastGoodPoint = gpuTracker.currentPoint
                    gpuPoint = lastPoint
                }
                
                let shouldAttemptVision =
                    !visionPermanentlyDisabled &&
                    (visionFallbackFramesRemaining <= 0 || frameCount % 6 == 0)
                if visionFallbackFramesRemaining > 0 {
                    visionFallbackFramesRemaining -= 1
                }

                if !shouldAttemptVision {
                    if let gpuPoint {
                        previousPoint = lastPoint
                        lastPoint = gpuPoint
                        lastBoundingBox = makeTrackingRect(
                            around: lastPoint,
                            width: normalizedBoxWidth,
                            height: normalizedBoxHeight,
                            fallback: lastBoundingBox
                        )
                        velocity = CGPoint(
                            x: lastPoint.x - previousPoint.x,
                            y: lastPoint.y - previousPoint.y
                        )
                    }
                } else {
                    do {
                        if request == nil {
                            request = makeVisionTrackingRequest(observation: VNDetectedObjectObservation(boundingBox: lastBoundingBox))
                        }
                        
                        guard let activeRequest = request else { return }
                        let visionOrientation = cgImageOrientation(from: preferredTransform)
                        try sequenceHandler.perform([activeRequest], on: pixelBuffer, orientation: visionOrientation)
                        
                        if let observation = activeRequest.results?.first as? VNDetectedObjectObservation,
                           observation.confidence > 0.08 {
                            let rect = observation.boundingBox
                            let visionPoint = CGPoint(
                                x: rect.midX,
                                y: 1 - rect.midY
                            )
                            // 始终用 Vision 返回的原始 observation 作为下一帧输入，
                            // 避免创建新的 VNDetectedObjectObservation 导致 tracker 泄漏
                            activeRequest.inputObservation = observation
                            lastBoundingBox = rect

                            let selectedPoint: CGPoint
                            if let gpuPoint {
                                let disagreement = hypot(visionPoint.x - gpuPoint.x, visionPoint.y - gpuPoint.y)
                                if disagreement > 0.10 {
                                    disagreementFrames += 1
                                    selectedPoint = gpuPoint
                                } else {
                                    disagreementFrames = max(0, disagreementFrames - 1)
                                    let visionWeight: CGFloat = disagreement < 0.02 ? 0.35 : 0.55
                                    selectedPoint = CGPoint(
                                        x: gpuPoint.x * (1 - visionWeight) + visionPoint.x * visionWeight,
                                        y: gpuPoint.y * (1 - visionWeight) + visionPoint.y * visionWeight
                                    )
                                }
                            } else {
                                disagreementFrames = 0
                                selectedPoint = visionPoint
                            }

                            previousPoint = lastPoint
                            lastPoint = selectedPoint
                            velocity = CGPoint(
                                x: lastPoint.x - previousPoint.x,
                                y: lastPoint.y - previousPoint.y
                            )
                            lostFrames = 0
                            visionErrorStreak = 0
                            if disagreementFrames >= 5, gpuPoint != nil {
                                visionFallbackFramesRemaining = max(visionFallbackFramesRemaining, 12)
                                if debugMode {
                                    print("Vision mismatched repeatedly, temporarily favoring GPU continuation.")
                                }
                            }
                        } else {
                            lostFrames += 1
                            if let gpuPoint {
                                previousPoint = lastPoint
                                lastPoint = gpuPoint
                                velocity = CGPoint(
                                    x: lastPoint.x - previousPoint.x,
                                    y: lastPoint.y - previousPoint.y
                                )
                            }
                            // 低置信时也不创建新 observation，保留上一帧的 observation 让 Vision 继续搜索
                            let velocityDecay = min(CGFloat(lostFrames) * 0.18, 0.85)
                            velocity.x *= (1 - velocityDecay)
                            velocity.y *= (1 - velocityDecay)
                            if debugMode, lostFrames == 1 {
                                print("Vision tracking lost, trying predicted reacquisition...")
                            }
                            if lostFrames >= 20, gpuPoint != nil {
                                // 长期丢失：重新初始化 Vision tracker，用当前 GPU 估计的位置
                                let expandedWidth = min(0.5, normalizedBoxWidth * 1.5)
                                let expandedHeight = min(0.5, normalizedBoxHeight * 1.5)
                                lastBoundingBox = makeTrackingRect(
                                    around: lastPoint,
                                    width: expandedWidth,
                                    height: expandedHeight,
                                    fallback: lastBoundingBox
                                )
                                if debugMode {
                                    print("Vision lost \(lostFrames) frames — reinitializing tracker with expanded box \(expandedWidth)x\(expandedHeight) at \(lastPoint)")
                                }
                                request = makeVisionTrackingRequest(
                                    observation: VNDetectedObjectObservation(boundingBox: lastBoundingBox)
                                )
                                lostFrames = 0
                                visionFallbackFramesRemaining = max(visionFallbackFramesRemaining, 6)
                            } else if lostFrames >= 10, gpuPoint != nil {
                                visionFallbackFramesRemaining = max(visionFallbackFramesRemaining, 12)
                                if debugMode {
                                    print("Vision lost too long, temporarily favoring GPU continuation.")
                                }
                            }
                        }
                    } catch {
                        visionErrorStreak += 1
                        let nsError = error as NSError
                        let trackerLimitHit =
                            nsError.domain == "com.apple.Vision" &&
                            nsError.code == 9 &&
                            error.localizedDescription.localizedCaseInsensitiveContains("Exceeded maximum allowed number of Trackers")
                        let boundingBoxError =
                            error.localizedDescription.localizedCaseInsensitiveContains("unexpected tracked object bounding box size") ||
                            error.localizedDescription.localizedCaseInsensitiveContains("bounding box")
                        if trackerLimitHit {
                            visionPermanentlyDisabled = true
                            visionFallbackFramesRemaining = Int.max / 4
                            request = nil
                        } else if boundingBoxError && visionErrorStreak >= 2 {
                            // bounding box 尺寸异常：旧 tracker 已经失效，
                            // 用当前 lastPoint 重建一个更大的搜索框重新初始化
                            let expandedWidth = min(0.5, normalizedBoxWidth * 1.5)
                            let expandedHeight = min(0.5, normalizedBoxHeight * 1.5)
                            lastBoundingBox = makeTrackingRect(
                                around: lastPoint,
                                width: expandedWidth,
                                height: expandedHeight,
                                fallback: lastBoundingBox
                            )
                            request = makeVisionTrackingRequest(
                                observation: VNDetectedObjectObservation(boundingBox: lastBoundingBox)
                            )
                            visionErrorStreak = 0
                            visionFallbackFramesRemaining = max(visionFallbackFramesRemaining, 6)
                            if debugMode {
                                print("Vision bounding box error — reinitializing tracker with expanded box \(expandedWidth)x\(expandedHeight) at \(lastPoint)")
                            }
                        } else {
                            // 不要销毁 request，只暂时回退到 GPU；
                            // 销毁 request 会导致下次创建新 tracker 泄漏资源
                            visionFallbackFramesRemaining = max(visionFallbackFramesRemaining, visionErrorStreak >= 3 ? 90 : 30)
                        }
                        if let gpuPoint {
                            previousPoint = lastPoint
                            lastPoint = gpuPoint
                            velocity = CGPoint(
                                x: lastPoint.x - previousPoint.x,
                                y: lastPoint.y - previousPoint.y
                            )
                        }
                        if debugMode {
                            if trackerLimitHit {
                                print("Vision tracker resource limit hit, switching to GPU-only continuation.")
                            }
                            print("Vision tracking error [\(visionErrorStreak)]: \(error)")
                        }
                    }
                }

                if shouldAttemptVision == false {
                    disagreementFrames = 0
                }
                
                trajectory.append(lastPoint)
                previousTexture = currentTexture
                frameCount += 1
                if frameCount == 1 || frameCount == totalFrames || frameCount % visionProgressInterval == 0 {
                    let prog = 0.05 + Double(frameCount) / Double(max(totalFrames, 1)) * 0.4
                    progressHandler(min(prog, 0.45), "Vision 跟踪中 \(frameCount)/\(totalFrames)")
                }
            }
        }
        
        return trajectory.isEmpty ? nil : trajectory
    }

    private func expandBoundingBox(_ rect: CGRect, scale: CGFloat) -> CGRect {
        let newWidth = min(0.5, max(rect.width, rect.width * scale))
        let newHeight = min(0.5, max(rect.height, rect.height * scale))
        let centerX = rect.midX
        let centerY = rect.midY
        let x = max(0, min(1 - newWidth, centerX - newWidth / 2))
        let y = max(0, min(1 - newHeight, centerY - newHeight / 2))
        return CGRect(x: x, y: y, width: newWidth, height: newHeight)
    }

    private func predictedSearchRect(
        from rect: CGRect,
        point: CGPoint,
        velocity: CGPoint,
        lostFrames: Int
    ) -> CGRect {
        let projectedCenterX = point.x + velocity.x * CGFloat(min(lostFrames, 6))
        let projectedCenterY = 1 - point.y - velocity.y * CGFloat(min(lostFrames, 6))
        let scale = min(2.8, 1.0 + CGFloat(lostFrames) * 0.2)
        let expanded = expandBoundingBox(rect, scale: scale)
        let width = expanded.width
        let height = expanded.height
        let x = max(0, min(1 - width, projectedCenterX - width / 2))
        let y = max(0, min(1 - height, projectedCenterY - height / 2))
        return CGRect(x: x, y: y, width: width, height: height)
    }

    private func makeTrackingRect(
        around point: CGPoint,
        width: CGFloat,
        height: CGFloat,
        fallback: CGRect
    ) -> CGRect {
        let x = max(0, min(1 - width, point.x - width / 2))
        let y = max(0, min(1 - height, 1 - point.y - height / 2))
        guard x.isFinite, y.isFinite, width.isFinite, height.isFinite, width > 0, height > 0 else {
            return fallback
        }
        let candidate = CGRect(x: x, y: y, width: width, height: height)
        return candidate
    }

    private func makeVisionTrackingRequest(
        observation: VNDetectedObjectObservation
    ) -> VNTrackObjectRequest {
        let request = VNTrackObjectRequest(detectedObjectObservation: observation)
        request.trackingLevel = .accurate
        return request
    }

    private struct VideoInfo {
        let track: AVAssetTrack
        let audioTrack: AVAssetTrack?
        let naturalSize: CGSize
        let preferredTransform: CGAffineTransform
        let duration: CMTime
        let nominalFrameRate: Float
    }

    private func loadVideoInfo(for asset: AVAsset) -> VideoInfo? {
        let semaphore = DispatchSemaphore(value: 0)
        var result: VideoInfo?

        Task {
            defer { semaphore.signal() }
            do {
                let tracks = try await asset.loadTracks(withMediaType: .video)
                guard let track = tracks.first else { return }
                let audioTrack = try await asset.loadTracks(withMediaType: .audio).first
                let naturalSize = try await track.load(.naturalSize)
                let preferredTransform = try await track.load(.preferredTransform)
                let duration = try await asset.load(.duration)
                let nominalFrameRate = try await track.load(.nominalFrameRate)
                result = VideoInfo(
                    track: track,
                    audioTrack: audioTrack,
                    naturalSize: naturalSize,
                    preferredTransform: preferredTransform,
                    duration: duration,
                    nominalFrameRate: nominalFrameRate
                )
            } catch {
                if debugMode {
                    print("加载视频信息失败: \(error)")
                }
            }
        }

        semaphore.wait()
        return result
    }
    
    // GPU 加速渲染 (Metal CIContext)
    private func applyGPULegacy(
        input: CVPixelBuffer,
        width: Int, height: Int,
        outputW: Int, outputH: Int,
        shiftX: CGFloat, shiftY: CGFloat,
        scale: CGFloat,
        pixelBufferPool: CVPixelBufferPool? = nil
    ) -> CVPixelBuffer? {
        var outBuf: CVPixelBuffer?
        if let pool = pixelBufferPool {
            guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &outBuf) == kCVReturnSuccess,
                  let _ = outBuf else { return nil }
        } else {
            let attrs: [String: Any] = [
                kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                kCVPixelBufferMetalCompatibilityKey as String: true
            ]
            guard CVPixelBufferCreate(
                kCFAllocatorDefault, outputW, outputH,
                kCVPixelFormatType_32BGRA,
                attrs as CFDictionary, &outBuf
            ) == kCVReturnSuccess, let _ = outBuf else { return nil }
        }
        let out = outBuf!

        let ciImage = CIImage(cvPixelBuffer: input)
        let center = CGPoint(x: CGFloat(width) / 2, y: CGFloat(height) / 2)
        let translated = ciImage.transformed(
            by: CGAffineTransform(translationX: shiftX, y: shiftY)
        )
        let scaled = translated.transformed(
            by: CGAffineTransform(translationX: center.x, y: center.y)
                .scaledBy(x: scale, y: scale)
                .translatedBy(x: -center.x, y: -center.y)
        )
        let cropped = scaled
            .clampedToExtent()
            .cropped(to: CGRect(x: 0, y: 0, width: outputW, height: outputH))

        ciContext.render(
            cropped,
            to: out,
            bounds: CGRect(x: 0, y: 0, width: outputW, height: outputH),
            colorSpace: CGColorSpaceCreateDeviceRGB()
        )

        return out
    }

    private func adjustedPresentationTime(for time: CMTime, relativeTo start: CMTime) -> CMTime {
        let adjusted = CMTimeSubtract(time, start)
        return CMTimeCompare(adjusted, .zero) < 0 ? .zero : adjusted
    }

    private func muxOriginalAudioIfPossible(
        from asset: AVAsset,
        into outputURL: URL,
        clipTimeRange: CMTimeRange,
        outputFileType: AVFileType
    ) -> Bool {
        let semaphore = DispatchSemaphore(value: 0)
        var result = false

        Task {
            defer { semaphore.signal() }
            do {
                let processedAsset = AVURLAsset(url: outputURL)
                let processedVideoTrack = try await processedAsset.loadTracks(withMediaType: .video).first
                let sourceAudioTrack = try await asset.loadTracks(withMediaType: .audio).first
                guard let processedVideoTrack,
                      let sourceAudioTrack else {
                    result = true
                    return
                }

                let processedDuration = try await processedAsset.load(.duration)
                let audioDuration = CMTimeMinimum(processedDuration, clipTimeRange.duration)
                let audioRange = CMTimeRange(start: clipTimeRange.start, duration: audioDuration)

                let composition = AVMutableComposition()
                guard let compositionVideoTrack = composition.addMutableTrack(
                    withMediaType: .video,
                    preferredTrackID: kCMPersistentTrackID_Invalid
                ),
                let compositionAudioTrack = composition.addMutableTrack(
                    withMediaType: .audio,
                    preferredTrackID: kCMPersistentTrackID_Invalid
                ) else {
                    return
                }

                try compositionVideoTrack.insertTimeRange(
                    CMTimeRange(start: .zero, duration: processedDuration),
                    of: processedVideoTrack,
                    at: .zero
                )
                compositionVideoTrack.preferredTransform = processedVideoTrack.preferredTransform

                try compositionAudioTrack.insertTimeRange(
                    audioRange,
                    of: sourceAudioTrack,
                    at: .zero
                )

                let tempURL = outputURL
                    .deletingLastPathComponent()
                    .appendingPathComponent(outputURL.deletingPathExtension().lastPathComponent + "_muxed")
                    .appendingPathExtension(outputURL.pathExtension)
                try? FileManager.default.removeItem(at: tempURL)

                guard let exporter = AVAssetExportSession(
                    asset: composition,
                    presetName: AVAssetExportPresetPassthrough
                ) else {
                    return
                }

                exporter.outputURL = tempURL
                exporter.outputFileType = outputFileType
                exporter.shouldOptimizeForNetworkUse = false

                await withCheckedContinuation { continuation in
                    exporter.exportAsynchronously {
                        continuation.resume()
                    }
                }

                guard exporter.status == .completed else {
                    if self.debugMode, let error = exporter.error {
                        print("音频合并失败: \(error)")
                    }
                    try? FileManager.default.removeItem(at: tempURL)
                    return
                }

                try? FileManager.default.removeItem(at: outputURL)
                try FileManager.default.moveItem(at: tempURL, to: outputURL)
                result = true
            } catch {
                if self.debugMode {
                    print("音频合并异常: \(error)")
                }
            }
        }

        semaphore.wait()
        return result
    }
}
