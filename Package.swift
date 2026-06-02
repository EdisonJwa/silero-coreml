// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SileroCoreML",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "SileroCoreML",
            targets: ["SileroCoreML"]
        ),
        .executable(
            name: "SileroVADExample",
            targets: ["SileroVADExample"]
        ),
        .executable(
            name: "SileroVADBenchmark",
            targets: ["SileroVADBenchmark"]
        ),
    ],
    targets: [
        .target(
            name: "SileroCoreML",
            resources: [.copy("Resources/SileroVADModel.mlpackage")]
        ),
        .executableTarget(
            name: "SileroVADExample",
            dependencies: ["SileroCoreML"]
        ),
        .executableTarget(
            name: "SileroVADBenchmark",
            dependencies: ["SileroCoreML"]
        ),
        .testTarget(
            name: "SileroCoreMLTests",
            dependencies: ["SileroCoreML"]
        ),
    ]
)
