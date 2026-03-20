// swift-tools-version: 5.9
// apple-bottom — BLASphemy
// Copyright 2026 Technology Residue
// Author: Grant David Heileman, Ph.D.

import PackageDescription

let package = Package(
    name: "AppleBottom",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "AppleBottom",
            targets: ["AppleBottom"]
        ),
    ],
    targets: [
        .target(
            name: "AppleBottom",
            path: "Sources/AppleBottom",
            resources: [
                .process("Kernels")
            ]
        ),
        .testTarget(
            name: "AppleBottomTests",
            dependencies: ["AppleBottom"],
            path: "Tests/AppleBottomTests"
        ),
        .executableTarget(
            name: "Benchmarks",
            dependencies: ["AppleBottom"],
            path: "Benchmarks"
        )
    ]
)
