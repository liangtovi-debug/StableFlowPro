// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "StableFlow",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "StableFlow",
            targets: ["StableFlow"]
        )
    ],
    targets: [
        .executableTarget(
            name: "StableFlow",
            dependencies: [],
            path: "Sources",
            resources: [
                .copy("Shaders.metal")
            ]
        ),
        .testTarget(
            name: "StableFlowTests",
            dependencies: ["StableFlow"],
            path: "Tests/StableFlowTests"
        )
    ]
)
