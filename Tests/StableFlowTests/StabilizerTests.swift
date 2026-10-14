import XCTest
import AVFoundation
import CoreGraphics
@testable import StableFlow

final class StabilizerTests: XCTestCase {
    func testTrimRangeUsesRequestedFractions() throws {
        let range = try XCTUnwrap(ImprovedStabilizer.makeClipTimeRange(
            duration: CMTime(value: 6000, timescale: 600), clipStart: 0.2, clipEnd: 0.8
        ))
        XCTAssertEqual(CMTimeGetSeconds(range.start), 2, accuracy: 0.000001)
        XCTAssertEqual(CMTimeGetSeconds(range.duration), 6, accuracy: 0.000001)
        XCTAssertEqual(CMTimeGetSeconds(CMTimeRangeGetEnd(range)), 8, accuracy: 0.000001)
    }

    func testTrimUsesSubsecondPrecisionForCoarseSourceTimescale() throws {
        let range = try XCTUnwrap(ImprovedStabilizer.makeClipTimeRange(
            duration: CMTime(value: 10, timescale: 1), clipStart: 0.125, clipEnd: 0.875
        ))
        XCTAssertEqual(CMTimeGetSeconds(range.start), 1.25, accuracy: 0.000001)
        XCTAssertEqual(CMTimeGetSeconds(range.duration), 7.5, accuracy: 0.000001)
    }

    func testInvalidTrimRangesAreRejected() {
        let duration = CMTime(seconds: 10, preferredTimescale: 600)
        let ranges: [(Double, Double)] = [
            (-0.1, 1), (0, 1.1), (0.8, 0.2), (0.5, 0.5),
            (.nan, 1), (0, .nan), (-.infinity, 1), (0, .infinity)
        ]
        for (start, end) in ranges {
            XCTAssertNil(ImprovedStabilizer.makeClipTimeRange(
                duration: duration, clipStart: start, clipEnd: end
            ), "Unexpectedly accepted \(start)...\(end)")
        }
        for invalidDuration in [CMTime.zero, .invalid, .indefinite, .positiveInfinity,
                                CMTime(value: -1, timescale: 1)] {
            XCTAssertNil(ImprovedStabilizer.makeClipTimeRange(
                duration: invalidDuration, clipStart: 0, clipEnd: 1
            ))
        }
    }

    func testTrackingAndRenderingUseHalfOpenFrameBoundaries() throws {
        let range = try XCTUnwrap(ImprovedStabilizer.makeClipTimeRange(
            duration: CMTime(value: 1, timescale: 1), clipStart: 0.2, clipEnd: 0.8
        ))
        let samples = (0...30).map { CMTime(value: Int64($0), timescale: 30) }
        let included = samples.enumerated().filter {
            ImprovedStabilizer.includesPresentationTime($0.element, in: range)
        }.map(\.offset)
        XCTAssertEqual(included, Array(6..<24))
        XCTAssertFalse(ImprovedStabilizer.includesPresentationTime(.invalid, in: range))
        XCTAssertFalse(ImprovedStabilizer.includesPresentationTime(.indefinite, in: range))
        XCTAssertFalse(ImprovedStabilizer.includesPresentationTime(
            CMTime(value: -1, timescale: 30), in: range
        ))
    }

    func testSameInputAndOutputAreRejectedWithoutTouchingFile() throws {
        try withTemporaryDirectory { directory in
            let input = directory.appendingPathComponent("input.mp4")
            let original = Data("keep the original source".utf8)
            try original.write(to: input)
            let stabilizer = ImprovedStabilizer()
            var status = ""
            let success = stabilizer.stabilizeVideo(
                inputURL: input, outputURL: input, trackingPoint: CGPoint(x: 40, y: 40)
            ) { _, message in status = message }
            XCTAssertFalse(success)
            XCTAssertEqual(status, "错误：输入和输出不能是同一文件")
            XCTAssertEqual(try Data(contentsOf: input), original)
            XCTAssertNil(stabilizer.lastAnalysis)
        }
    }

    func testSymlinkToInputCannotBeUsedAsOutput() throws {
        try withTemporaryDirectory { directory in
            let input = directory.appendingPathComponent("input.mp4")
            let output = directory.appendingPathComponent("alias.mp4")
            let original = Data("keep the original source".utf8)
            try original.write(to: input)
            try FileManager.default.createSymbolicLink(at: output, withDestinationURL: input)
            var status = ""
            let success = ImprovedStabilizer().stabilizeVideo(
                inputURL: input, outputURL: output, trackingPoint: CGPoint(x: 40, y: 40)
            ) { _, message in status = message }
            XCTAssertFalse(success)
            XCTAssertEqual(status, "错误：输入和输出不能是同一文件")
            XCTAssertEqual(try Data(contentsOf: input), original)
        }
    }

    func testFailedInputReadPreservesExistingDestination() throws {
        try withTemporaryDirectory { directory in
            let output = directory.appendingPathComponent("existing.mp4")
            let original = Data("previous successful export".utf8)
            try original.write(to: output)
            let stabilizer = ImprovedStabilizer()
            let success = stabilizer.stabilizeVideo(
                inputURL: directory.appendingPathComponent("missing.mp4"),
                outputURL: output, trackingPoint: CGPoint(x: 40, y: 40)
            ) { _, _ in }
            XCTAssertFalse(success)
            XCTAssertEqual(try Data(contentsOf: output), original)
            XCTAssertNil(stabilizer.lastAnalysis)
            XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), ["existing.mp4"])
        }
    }

    func testNonfiniteProcessingParametersAreRejectedBeforeReading() throws {
        try withTemporaryDirectory { directory in
            var status = ""
            let success = ImprovedStabilizer().stabilizeVideo(
                inputURL: directory.appendingPathComponent("missing.mp4"),
                outputURL: directory.appendingPathComponent("output.mp4"),
                trackingPoint: CGPoint(x: CGFloat.nan, y: 40)
            ) { _, message in status = message }
            XCTAssertFalse(success)
            XCTAssertEqual(status, "错误：无效处理参数")
        }
    }

    func testTemplateMatchingUsesActualStrideAndEntireSearchRadius() {
        let width = 51
        let height = 51
        let size = 11
        let pixels = texturedPixels(count: width * height)
        for (x, y) in [(0, 0), (20, 20), (40, 40), (40, 0), (0, 40)] {
            let template = patch(pixels, width: width, x: x, y: y, size: size)
            XCTAssertEqual(FeatureMotionEstimator.matchTemplate(
                template, size: size, in: pixels, width: width, height: height
            ), CGPoint(x: x + size / 2, y: y + size / 2))
        }
    }

    func testTemplateMatchingHandlesClippedNonsquareSearchRegion() {
        let width = 33
        let height = 28
        let size = 11
        let pixels = texturedPixels(count: width * height)
        let template = patch(pixels, width: width, x: 22, y: 17, size: size)
        XCTAssertEqual(FeatureMotionEstimator.matchTemplate(
            template, size: size, in: pixels, width: width, height: height
        ), CGPoint(x: 27, y: 22))
        XCTAssertNil(FeatureMotionEstimator.matchTemplate(
            template, size: size, in: Array(pixels.dropLast()), width: width, height: height
        ))
        XCTAssertNil(FeatureMotionEstimator.matchTemplate(
            [Float](repeating: 0, count: size * size), size: size,
            in: pixels, width: width, height: height
        ))
    }

    func testMotionInliersKeepDisplacementComponentsPaired() {
        let displacements = [
            CGPoint(x: -25, y: 1), CGPoint(x: 1, y: -25),
            CGPoint(x: 0, y: 0), CGPoint(x: 2, y: 2), CGPoint(x: 25, y: 25)
        ]
        XCTAssertEqual(FeatureMotionEstimator.translationInliers(displacements), [
            CGPoint(x: 0, y: 0), CGPoint(x: 2, y: 2)
        ])
        XCTAssertEqual(FeatureMotionEstimator.translationInliers([]), [])
    }

    private func texturedPixels(count: Int) -> [Float] {
        var state: UInt32 = 42
        return (0..<count).map { _ in
            state = state &* 1_664_525 &+ 1_013_904_223
            return Float((state >> 24) & 255)
        }
    }

    private func patch(_ pixels: [Float], width: Int, x: Int, y: Int, size: Int) -> [Float] {
        (0..<size).flatMap { row in
            (0..<size).map { column in pixels[(y + row) * width + x + column] }
        }
    }

    private func withTemporaryDirectory(_ body: (URL) throws -> Void) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try body(directory)
    }
}
