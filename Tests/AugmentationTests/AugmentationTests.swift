import XCTest
@testable import Augmentation

import TensorFlow

final class AugmentationTests: XCTestCase {
    func testJitter() {
        let ones = Tensor<Float>(ones: [24, 3, 32, 32])
        XCTAssertEqual(jitter(ones), ones)
        let mixed = Tensor<Float>(ones: [24, 3, 16, 16]).concatenated(with: Tensor<Float>(zeros: [24, 3, 16, 16]))
        XCTAssertEqual(jitter(mixed), mixed)
        let alternating = Tensor<Float>([[[[0,1]]]]).tiled(multiples: [128, 3, 32, 16])
        XCTAssertNotEqual(jitter(alternating), alternating)
    }

    static var allTests = [
        ("testExample", testJitter),
    ]
}
