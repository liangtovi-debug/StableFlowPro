import Foundation
import Metal
import XCTest
@testable import StableFlow

final class GPUStabilizerTests: XCTestCase {
    private func makeTracker() throws -> GPUStabilizer {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("A Metal device is required for GPU tracking regression tests")
        }
        let tracker = try XCTUnwrap(GPUStabilizer())
        // Compile the checked-in source explicitly so these tests also work when
        // SwiftPM has copied the .metal resource rather than built a metallib.
        let sourceURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/Shaders.metal")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)
        let library = try tracker.device.makeLibrary(source: source, options: nil)
        tracker.opticalFlowPipeline = try tracker.device.makeComputePipelineState(
            function: XCTUnwrap(library.makeFunction(name: "opticalFlowGradient")))
        tracker.opticalFlowMultiPointPipeline = try tracker.device.makeComputePipelineState(
            function: XCTUnwrap(library.makeFunction(name: "opticalFlowMultiPoint")))
        tracker.templateMatchPipeline = try tracker.device.makeComputePipelineState(
            function: XCTUnwrap(library.makeFunction(name: "templateMatchParallel")))
        tracker.extractPatchPipeline = try tracker.device.makeComputePipelineState(
            function: XCTUnwrap(library.makeFunction(name: "extractPatch")))
        return tracker
    }

    private func noise(width: Int, height: Int) -> [UInt8] {
        var state: UInt32 = 0x12345678
        return (0..<(width * height)).map { _ in
            state ^= state << 13
            state ^= state >> 17
            state ^= state << 5
            return UInt8(truncatingIfNeeded: state)
        }
    }

    private func translated(_ pixels: [UInt8], width: Int, height: Int, dx: Int, dy: Int) -> [UInt8] {
        var result = [UInt8](repeating: 0, count: pixels.count)
        for y in 0..<height {
            for x in 0..<width {
                let sourceX = x - dx
                let sourceY = y - dy
                if sourceX >= 0 && sourceX < width && sourceY >= 0 && sourceY < height {
                    result[y * width + x] = pixels[sourceY * width + sourceX]
                }
            }
        }
        return result
    }

    private func texture(_ pixels: [UInt8], width: Int, height: Int, tracker: GPUStabilizer) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm, width: width, height: height, mipmapped: false)
        descriptor.storageMode = tracker.device.hasUnifiedMemory ? .shared : .managed
        descriptor.usage = [.shaderRead, .shaderWrite]
        let texture = try XCTUnwrap(tracker.device.makeTexture(descriptor: descriptor))
        pixels.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: bytes.baseAddress!,
                bytesPerRow: width)
        }
        return texture
    }

    func testBothFlowKernelsTrackSignedTranslation() throws {
        let tracker = try makeTracker()
        let side = 128
        let pixels = noise(width: side, height: side)
        let prev = try texture(pixels, width: side, height: side, tracker: tracker)
        let curr = try texture(
            translated(pixels, width: side, height: side, dx: 3, dy: -2),
            width: side, height: side, tracker: tracker)
        let point = SIMD2<Float>(0.5, 0.5)
        let serial = try XCTUnwrap(tracker.trackPointWithGradient(prev: prev, curr: curr, point: point))
        let parallel = try XCTUnwrap(tracker.trackPointMultiPointParallel(prev: prev, curr: curr, centerPoint: point))
        XCTAssertEqual(serial.offset.x, 3, accuracy: 0.001)
        XCTAssertEqual(serial.offset.y, -2, accuracy: 0.001)
        XCTAssertEqual(serial.confidence, 1)
        XCTAssertEqual(parallel.x, 3, accuracy: 0.001)
        XCTAssertEqual(parallel.y, -2, accuracy: 0.001)
        tracker.opticalFlowMultiPointPipeline = nil
        let fallback = try XCTUnwrap(tracker.trackPointMultiPointParallel(prev: prev, curr: curr, centerPoint: point))
        XCTAssertEqual(fallback.x, 3, accuracy: 0.001)
        XCTAssertEqual(fallback.y, -2, accuracy: 0.001)
    }

    func testFirstRecoveryCallTracksCurrentFrame() throws {
        let tracker = try makeTracker()
        let side = 128
        let pixels = noise(width: side, height: side)
        let prev = try texture(pixels, width: side, height: side, tracker: tracker)
        let curr = try texture(
            translated(pixels, width: side, height: side, dx: 3, dy: -2),
            width: side, height: side, tracker: tracker)
        let result = tracker.trackWithRecovery(prev: prev, curr: curr, centerPoint: SIMD2<Float>(0.5, 0.5))
        XCTAssertNotNil(tracker.templateTexture)
        XCTAssertEqual((result.x - 0.5) * Float(side), 3 * 0.99, accuracy: 0.01)
        XCTAssertEqual((result.y - 0.5) * Float(side), -2 * 0.99, accuracy: 0.01)
    }

    func testTemplateSearchIncludesPositiveEdgesAndCorners() throws {
        let tracker = try makeTracker()
        let side = 256
        let pixels = noise(width: side, height: side)
        let prev = try texture(pixels, width: side, height: side, tracker: tracker)
        tracker.currentPoint = SIMD2<Float>(0.5, 0.5)
        tracker.initTemplate(from: prev, at: tracker.currentPoint)
        XCTAssertNotNil(tracker.templateTexture)

        for radius in [8, 16, 40] {
            tracker.searchRadiusTemplate = Float(radius)
            for (dx, dy) in [(radius, 0), (0, radius), (radius, radius), (-radius, -radius)] {
                let curr = try texture(
                    translated(pixels, width: side, height: side, dx: dx, dy: dy),
                    width: side, height: side, tracker: tracker)
                let result = try XCTUnwrap(tracker.templateMatch(texture: curr), "radius=\(radius), shift=(\(dx), \(dy))")
                XCTAssertEqual((result.position.x - 0.5) * Float(side), Float(dx), accuracy: 0.01)
                XCTAssertEqual((result.position.y - 0.5) * Float(side), Float(dy), accuracy: 0.01)
                XCTAssertEqual(result.error, 0, accuracy: 0.000001)
            }
        }
    }

    func testFlowRejectsPointWithoutGradientNeighborMargin() throws {
        let tracker = try makeTracker()
        let side = 128
        let pixels = noise(width: side, height: side)
        let frame = try texture(pixels, width: side, height: side, tracker: tracker)
        // radius 16 + window half-width 3 is still one pixel too close.
        let point = SIMD2<Float>(19 / Float(side), 0.5)
        XCTAssertNil(tracker.trackPointWithGradient(prev: frame, curr: frame, point: point))
        // The rightmost grid point is at x=19, so none has enough margin.
        let gridCenter = SIMD2<Float>(13 / Float(side), 0.5)
        XCTAssertNil(tracker.trackPointMultiPointParallel(prev: frame, curr: frame, centerPoint: gridCenter))
    }

    func testFlatFramesDoNotInventMotion() throws {
        let tracker = try makeTracker()
        let side = 128
        let pixels = [UInt8](repeating: 127, count: side * side)
        let frame = try texture(pixels, width: side, height: side, tracker: tracker)
        let center = SIMD2<Float>(0.5, 0.5)
        XCTAssertNil(tracker.trackPointWithGradient(prev: frame, curr: frame, point: center))
        XCTAssertNil(tracker.trackPointMultiPointParallel(prev: frame, curr: frame, centerPoint: center))
        for _ in 0..<3 {
            let result = tracker.trackWithRecovery(prev: frame, curr: frame, centerPoint: center)
            XCTAssertEqual(result.x, center.x, accuracy: 0.000001)
            XCTAssertEqual(result.y, center.y, accuracy: 0.000001)
        }
    }

    func testNoValidTemplateCandidateDoesNotReplaceLastGoodPoint() throws {
        let tracker = try makeTracker()
        let side = 16 // Smaller than the default 21-pixel template.
        let frame = try texture(noise(width: side, height: side), width: side, height: side, tracker: tracker)
        tracker.currentPoint = SIMD2<Float>(0.6, 0.6)
        tracker.lastGoodPoint = SIMD2<Float>(0.5, 0.5)
        tracker.initTemplate(from: frame, at: tracker.currentPoint)
        let originalTemplate = try XCTUnwrap(tracker.templateTexture)
        XCTAssertNil(tracker.templateMatch(texture: frame))
        _ = tracker.trackWithRecovery(prev: frame, curr: frame, centerPoint: tracker.currentPoint)
        XCTAssertEqual(tracker.lastGoodPoint, SIMD2<Float>(0.5, 0.5))
        XCTAssertTrue(tracker.templateTexture === originalTemplate)
    }
}
