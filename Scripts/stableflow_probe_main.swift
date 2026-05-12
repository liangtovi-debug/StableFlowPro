import Foundation
import CoreGraphics

@main
struct StableFlowProbeMain {
    static func main() {
        let args = CommandLine.arguments
        guard args.count >= 7 else {
            fputs("usage: StableFlowProbe <input> <output> <x> <y> <box_w> <box_h> [strength]\n", stderr)
            exit(2)
        }

        let inputURL = URL(fileURLWithPath: args[1])
        let outputURL = URL(fileURLWithPath: args[2])
        let x = Double(args[3]) ?? 0
        let y = Double(args[4]) ?? 0
        let boxW = Double(args[5]) ?? 80
        let boxH = Double(args[6]) ?? 80
        let strength = args.count >= 8 ? (Double(args[7]) ?? 0.8) : 0.8

        let stabilizer = ImprovedStabilizer()
        stabilizer.debugMode = true
        let success = stabilizer.stabilizeVideo(
            inputURL: inputURL,
            outputURL: outputURL,
            trackingPoint: CGPoint(x: x, y: y),
            trackingBoxSize: CGSize(width: boxW, height: boxH),
            exportFormat: "mp4",
            smoothFactor: 0.5,
            stabilizeStrength: strength,
            clipStart: 0,
            clipEnd: 1
        ) { progress, status in
            print(String(format: "progress=%.3f status=%@", progress, status))
        }

        print("success=\(success)")
        if let analysis = stabilizer.lastAnalysis {
            print(
                String(
                    format: "analysis meanCenterError=%.6f maxCenterError=%.6f cropRatio=%.6f cropLimited=%@ meanTransformDelta=%.6f maxTransformDelta=%.6f",
                    analysis.meanCenterError,
                    analysis.maxCenterError,
                    analysis.cropRatio,
                    analysis.cropLimited ? "true" : "false",
                    analysis.meanTransformDelta,
                    analysis.maxTransformDelta
                )
            )
            let preview = analysis.transforms.prefix(5).map {
                String(format: "(%.2f,%.2f)", $0.x, $0.y)
            }.joined(separator: ", ")
            print("transforms \(preview)")
        } else {
            print("analysis unavailable")
        }
    }
}
