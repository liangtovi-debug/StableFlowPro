import Foundation
import Metal
import MetalKit
import CoreVideo

class MetalStabilizer {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let analyzePipeline: MTLComputePipelineState
    private let stabilizePipeline: MTLComputePipelineState
    private let textureCache: CVMetalTextureCache
    
    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            print("Metal: Failed to create device or queue")
            return nil
        }
        
        self.device = device
        self.commandQueue = queue
        
        // 加载 Metal 库
        guard let library = device.makeDefaultLibrary(),
              let analyzeFunc = library.makeFunction(name: "analyzeMotion"),
              let stabilizeFunc = library.makeFunction(name: "stabilizeFrame") else {
            print("Metal: Failed to load shader functions")
            return nil
        }
        
        do {
            analyzePipeline = try device.makeComputePipelineState(function: analyzeFunc)
            stabilizePipeline = try device.makeComputePipelineState(function: stabilizeFunc)
        } catch {
            print("Metal pipeline error: \(error)")
            return nil
        }
        
        // 创建纹理缓存
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        guard let textureCache = cache else {
            print("Metal: Failed to create texture cache")
            return nil
        }
        self.textureCache = textureCache
    }
    
    // 分析两帧之间的运动
    func analyzeMotion(prev: CVPixelBuffer, curr: CVPixelBuffer) -> (dx: Float, dy: Float) {
        guard let prevTex = createTexture(from: prev),
              let currTex = createTexture(from: curr) else {
            return (0, 0)
        }
        
        // 创建输出缓冲区
        var sumX: UInt32 = 0
        var sumY: UInt32 = 0
        var totalWeight: UInt32 = 0
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return (0, 0)
        }
        
        computeEncoder.setComputePipelineState(analyzePipeline)
        computeEncoder.setTexture(prevTex, index: 0)
        computeEncoder.setTexture(currTex, index: 1)
        computeEncoder.setBytes(&sumX, length: MemoryLayout<UInt32>.size, index: 0)
        computeEncoder.setBytes(&sumY, length: MemoryLayout<UInt32>.size, index: 1)
        computeEncoder.setBytes(&totalWeight, length: MemoryLayout<UInt32>.size, index: 2)
        
        let w = prevTex.width
        let h = prevTex.height
        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (w + threadGroupSize.width - 1) / threadGroupSize.width,
            height: (h + threadGroupSize.height - 1) / threadGroupSize.height,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let weight = Float(totalWeight) / 1000.0
        if weight > 0.1 {
            let dx = Float(sumX) / weight / 1000.0
            let dy = Float(sumY) / weight / 1000.0
            return (dx, dy)
        }
        
        return (0, 0)
    }
    
    // 稳定帧
    func stabilizeFrame(input: CVPixelBuffer, output: CVPixelBuffer, offset: CGPoint, cropOrigin: CGPoint) {
        guard let inputTex = createTexture(from: input),
              let outputTex = createTexture(from: output) else {
            return
        }
        
        var offsetX = Float(offset.x)
        var offsetY = Float(offset.y)
        var cropX = Float(cropOrigin.x)
        var cropY = Float(cropOrigin.y)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        computeEncoder.setComputePipelineState(stabilizePipeline)
        computeEncoder.setTexture(inputTex, index: 0)
        computeEncoder.setTexture(outputTex, index: 1)
        
        withUnsafePointer(to: &offsetX) { ptr in
            computeEncoder.setBytes(ptr, length: MemoryLayout<Float>.size, index: 0)
        }
        withUnsafePointer(to: &offsetY) { ptr in
            computeEncoder.setBytes(ptr, length: MemoryLayout<Float>.size, index: 1)
        }
        withUnsafePointer(to: &cropX) { ptr in
            computeEncoder.setBytes(ptr, length: MemoryLayout<Float>.size, index: 2)
        }
        withUnsafePointer(to: &cropY) { ptr in
            computeEncoder.setBytes(ptr, length: MemoryLayout<Float>.size, index: 3)
        }
        
        let w = outputTex.width
        let h = outputTex.height
        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (w + threadGroupSize.width - 1) / threadGroupSize.width,
            height: (h + threadGroupSize.height - 1) / threadGroupSize.height,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
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
}
