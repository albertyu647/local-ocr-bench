// Baseline: macOS built-in Vision OCR (VNRecognizeTextRequest, accurate mode).
// Usage: swift apple_vision_ocr.swift <image> -> prints text lines top-to-bottom.
import Foundation
import Vision
import AppKit

let path = CommandLine.arguments[1]
guard let img = NSImage(contentsOfFile: path),
      let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    fputs("cannot load \(path)\n", stderr); exit(1)
}
let req = VNRecognizeTextRequest()
req.recognitionLevel = .accurate
req.usesLanguageCorrection = true
req.recognitionLanguages = ["en-US"]
try VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
let obs = (req.results ?? []).sorted {
    // top-to-bottom, then left-to-right (Vision's origin is bottom-left)
    abs($0.boundingBox.midY - $1.boundingBox.midY) > 0.01 ? $0.boundingBox.midY > $1.boundingBox.midY
                                                           : $0.boundingBox.minX < $1.boundingBox.minX
}
for o in obs { if let t = o.topCandidates(1).first { print(t.string) } }
