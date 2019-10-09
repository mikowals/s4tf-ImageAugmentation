import XCTest
@testable import Augmentation

final class AugmentationTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(Augmentation().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
