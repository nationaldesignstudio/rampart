// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "RampartCoreML",
    platforms: [
        .iOS(.v17),
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "RampartCoreML",
            targets: ["RampartCoreML"]
        ),
        .executable(
            name: "RampartCLI",
            targets: ["RampartCLI"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/weichsel/ZIPFoundation.git", from: "0.9.20")
    ],
    targets: [
        .target(
            name: "RampartCoreML",
            dependencies: [
                .product(name: "ZIPFoundation", package: "ZIPFoundation")
            ]
        ),
        .executableTarget(
            name: "RampartCLI",
            dependencies: ["RampartCoreML"]
        ),
        .testTarget(
            name: "RampartCoreMLTests",
            dependencies: [
                "RampartCoreML",
                .product(name: "ZIPFoundation", package: "ZIPFoundation")
            ]
        )
    ]
)
