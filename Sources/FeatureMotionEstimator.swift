import Foundation
import Metal
import MetalKit
import CoreVideo
import Accelerate

/// 基于特征点匹配的运动估计器
class FeatureMotionEstimator {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let textureCache: CVMetalTextureCache
    
    // 运动历史 (用于平滑)
    private var motionHistory: [(dx: Float, dy: Float)] = []
    private let maxHistory = 10
    
    // 特征点网格
    private let gridSize = 8
    private var prevFeatures: [CGPoint] = []
    
    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            print("FeatureMotionEstimator: Failed to create device or queue")
            return nil
        }
        
        self.device = device
        self.commandQueue = queue
        
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        guard let textureCache = cache else {
            print("FeatureMotionEstimator: Failed to create texture cache")
            return nil
        }
        self.textureCache = textureCache
    }
    
    /// 估计两帧之间的运动 (使用特征点匹配)
    func estimateMotion(prev: CVPixelBuffer, curr: CVPixelBuffer) -> (dx: Float, dy: Float, confidence: Float) {
        guard let prevTex = createTexture(from: prev),
              let currTex = createTexture(from: curr) else {
            return (0, 0, 0)
        }
        
        // 1. 提取特征点
        let features = extractFeatures(from: prevTex)
        
        // 2. 在当前帧中追踪特征点
        let matches = trackFeatures(from: prevTex, to: currTex, features: features)
        
        // 3. 使用RANSAC计算运动变换
        let (dx, dy, confidence) = computeMotion(matches: matches)
        
        return (dx, dy, confidence)
    }
    
    /// 提取Shi-Tomasi角点特征点
    private func extractFeatures(from texture: MTLTexture) -> [CGPoint] {
        let width = texture.width
        let height = texture.height
        
        // 读取纹理数据
        let bytesPerRow = width * 4
        var data = [UInt8](repeating: 0, count: height * bytesPerRow)
        
        texture.getBytes(&data, bytesPerRow: bytesPerRow, from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0), size: MTLSize(width: width, height: height, depth: 1)), mipmapLevel: 0)
        
        // 转换为灰度图用于特征检测
        var grayscale = [Float](repeating: 0, count: width * height)
        for i in 0..<(width * height) {
            let b = Float(data[i * 4])
            let g = Float(data[i * 4 + 1])
            let r = Float(data[i * 4 + 2])
            grayscale[i] = r * 0.299 + g * 0.587 + b * 0.114
        }
        
        // 计算梯度
        var gradX = [Float](repeating: 0, count: width * height)
        var gradY = [Float](repeating: 0, count: width * height)
        
        for y in 1..<(height-1) {
            for x in 1..<(width-1) {
                let idx = y * width + x
                gradX[idx] = grayscale[idx + 1] - grayscale[idx - 1]
                gradY[idx] = grayscale[idx + width] - grayscale[idx - width]
            }
        }
        
        // Harris角点响应
        var corners: [(x: Int, y: Int, response: Float)] = []
        let windowSize = 5
        let k: Float = 0.04 // Harris常数
        
        for y in windowSize..<(height-windowSize) {
            for x in windowSize..<(width-windowSize) {
                // 计算M矩阵元素
                var Ixx: Float = 0
                var Iyy: Float = 0
                var Ixy: Float = 0
                
                for wy in -windowSize/2...windowSize/2 {
                    for wx in -windowSize/2...windowSize/2 {
                        let idx = (y + wy) * width + (x + wx)
                        let gx = gradX[idx]
                        let gy = gradY[idx]
                        Ixx += gx * gx
                        Iyy += gy * gy
                        Ixy += gx * gy
                    }
                }
                
                // Harris响应
                let det = Ixx * Iyy - Ixy * Ixy
                let trace = Ixx + Iyy
                let response = det - k * trace * trace
                
                if response > 1000 { // 阈值
                    corners.append((x, y, response))
                }
            }
        }
        
        // 按响应值排序并选择最好的特征点
        corners.sort { $0.response > $1.response }
        
        // 使用网格确保特征点分布均匀
        var features: [CGPoint] = []
        let cellW = width / gridSize
        let cellH = height / gridSize
        var occupiedCells = Set<String>()
        
        for corner in corners {
            let cellX = corner.x / cellW
            let cellY = corner.y / cellH
            let cellKey = "\(cellX)-\(cellY)"
            
            // 检查该网格是否已有特征点
            let hasNearby = features.contains { f in
                let cf = CGPoint(x: corner.x, y: corner.y)
                let dist = hypot(f.x - cf.x, f.y - cf.y)
                return dist < 30
            }
            
            if !hasNearby && !occupiedCells.contains(cellKey) && features.count < 50 {
                features.append(CGPoint(x: CGFloat(corner.x), y: CGFloat(corner.y)))
                occupiedCells.insert(cellKey)
            }
            
            if features.count >= 50 { break }
        }
        
        prevFeatures = features
        return features
    }
    
    /// 追踪特征点 (使用模板匹配)
    private func trackFeatures(from prevTex: MTLTexture, to currTex: MTLTexture, features: [CGPoint]) -> [(prev: CGPoint, curr: CGPoint)] {
        var matches: [(prev: CGPoint, curr: CGPoint)] = []
        
        let searchRadius: CGFloat = 20
        let templateSize = 11
        
        for feature in features {
            if let match = findMatch(from: prevTex, to: currTex, point: feature, searchRadius: searchRadius, templateSize: templateSize) {
                matches.append((prev: feature, curr: match))
            }
        }
        
        return matches
    }
    
    /// 在目标帧中查找最佳匹配
    private func findMatch(from prevTex: MTLTexture, to currTex: MTLTexture, point: CGPoint, searchRadius: CGFloat, templateSize: Int) -> CGPoint? {
        // 读取像素数据
        var prevData = readRegion(texture: prevTex, x: Int(point.x) - templateSize/2, y: Int(point.y) - templateSize/2, width: templateSize, height: templateSize)
        let currData = readRegion(texture: currTex, x: Int(point.x) - Int(searchRadius) - templateSize/2, y: Int(point.y) - Int(searchRadius) - templateSize/2, width: Int(searchRadius * 2) + templateSize, height: Int(searchRadius * 2) + templateSize)
        
        guard !prevData.isEmpty && !currData.isEmpty else { return nil }
        
        // 归一化模板
        normalizeTemplate(&prevData)
        
        // 滑动窗口搜索
        var bestMatch = CGPoint.zero
        var bestScore: Float = .greatestFiniteMagnitude // SSD，越小越好
        
        let searchW = Int(searchRadius * 2) + 1
        
        for sy in 0..<(searchW - templateSize + 1) {
            for sx in 0..<(searchW - templateSize + 1) {
                // 提取搜索窗口中的模板
                var searchTemplate = [Float](repeating: 0, count: templateSize * templateSize)
                
                for ty in 0..<templateSize {
                    for tx in 0..<templateSize {
                        let srcIdx = (sy + ty) * searchW + (sx + tx)
                        let dstIdx = ty * templateSize + tx
                        if srcIdx < currData.count && dstIdx < searchTemplate.count {
                            searchTemplate[dstIdx] = currData[srcIdx]
                        }
                    }
                }
                
                normalizeTemplate(&searchTemplate)
                
                // 计算SSD
                var ssd: Float = 0
                for i in 0..<min(prevData.count, searchTemplate.count) {
                    let diff = prevData[i] - searchTemplate[i]
                    ssd += diff * diff
                }
                
                if ssd < bestScore {
                    bestScore = ssd
                    bestMatch = CGPoint(
                        x: point.x + CGFloat(sx - Int(searchRadius)),
                        y: point.y + CGFloat(sy - Int(searchRadius))
                    )
                }
            }
        }
        
        // SSD阈值检查
        if bestScore < 0.3 {
            return bestMatch
        }
        
        return nil
    }
    
    /// 读取纹理区域
    private func readRegion(texture: MTLTexture, x: Int, y: Int, width: Int, height: Int) -> [Float] {
        let w = texture.width
        let h = texture.height
        
        // 边界检查
        let rx = max(0, x)
        let ry = max(0, y)
        let rw = min(width, w - rx)
        let rh = min(height, h - ry)
        
        guard rw > 0 && rh > 0 else { return [] }
        
        let bytesPerRow = rw * 4
        var data = [UInt8](repeating: 0, count: rh * bytesPerRow)
        
        texture.getBytes(&data, bytesPerRow: bytesPerRow, from: MTLRegion(origin: MTLOrigin(x: rx, y: ry, z: 0), size: MTLSize(width: rw, height: rh, depth: 1)), mipmapLevel: 0)
        
        // 转换为灰度
        var grayscale = [Float](repeating: 0, count: rw * rh)
        for i in 0..<(rw * rh) {
            let b = Float(data[i * 4])
            let g = Float(data[i * 4 + 1])
            let r = Float(data[i * 4 + 2])
            grayscale[i] = r * 0.299 + g * 0.587 + b * 0.114
        }
        
        return grayscale
    }
    
    /// 归一化模板
    private func normalizeTemplate(_ template: inout [Float]) {
        var mean: Float = 0
        var std: Float = 0
        
        for v in template { mean += v }
        mean /= Float(template.count)
        
        for v in template {
            let diff = v - mean
            std += diff * diff
        }
        std = sqrt(std / Float(template.count))
        
        if std > 0.001 {
            for i in 0..<template.count {
                template[i] = (template[i] - mean) / std
            }
        }
    }
    
    /// 使用RANSAC计算运动变换
    private func computeMotion(matches: [(prev: CGPoint, curr: CGPoint)]) -> (dx: Float, dy: Float, confidence: Float) {
        guard matches.count >= 5 else {
            return (0, 0, 0)
        }
        
        // 简单方法：计算所有匹配点的中位位移
        var dxList: [Float] = []
        var dyList: [Float] = []
        
        for m in matches {
            dxList.append(Float(m.curr.x - m.prev.x))
            dyList.append(Float(m.curr.y - m.prev.y))
        }
        
        // 使用中位数过滤异常值
        dxList.sort()
        dyList.sort()
        
        let medianIdx = dxList.count / 2
        var dx = dxList[medianIdx]
        var dy = dyList[medianIdx]
        
        // 计算与中位数的偏差，过滤异常值
        var filteredDx: [Float] = []
        var filteredDy: [Float] = []
        
        for i in 0..<dxList.count {
            if abs(dxList[i] - dx) < 20 && abs(dyList[i] - dy) < 20 {
                filteredDx.append(dxList[i])
                filteredDy.append(dyList[i])
            }
        }
        
        guard filteredDx.count >= 3 else {
            return (dx, dy, Float(matches.count) / 50.0)
        }
        
        // 计算最终运动
        dx = filteredDx.reduce(0, +) / Float(filteredDx.count)
        dy = filteredDy.reduce(0, +) / Float(filteredDy.count)
        
        let confidence = Float(filteredDx.count) / Float(matches.count)
        
        // 平滑处理
        motionHistory.append((dx, dy))
        if motionHistory.count > maxHistory {
            motionHistory.removeFirst()
        }
        
        // 加权平均 (重视最近的)
        var smoothDx: Float = 0
        var smoothDy: Float = 0
        var weight: Float = 0
        
        for (i, m) in motionHistory.enumerated() {
            let w = Float(i + 1)
            smoothDx += m.dx * w
            smoothDy += m.dy * w
            weight += w
        }
        
        return (smoothDx / weight, smoothDy / weight, confidence)
    }
    
    private func createTexture(from pixelBuffer: CVPixelBuffer) -> MTLTexture? {
        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)
        
        var cvTex: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(
            nil, textureCache, pixelBuffer, nil,
            .bgra8Unorm, w, h, 0, &cvTex
        )
        
        guard let cvTex = cvTex else { return nil }
        return CVMetalTextureGetTexture(cvTex)
    }
    
    /// 重置运动历史
    func reset() {
        motionHistory.removeAll()
        prevFeatures.removeAll()
    }
}
