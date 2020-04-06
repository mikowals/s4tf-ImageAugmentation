import TensorFlow

public func shuffle<T, R>(tuple: (Tensor<T>, Tensor<R>)) -> (Tensor<T>, Tensor<R>) {
    let batchSize = tuple.0.shape[0]
    let shuffledIndices = _Raw.randomShuffle(value: Tensor<Int32>(rangeFrom: 0, to: Int32(batchSize), stride: 1))
    let labels = _Raw.gatherV2(params: tuple.0, indices: shuffledIndices, axis: Tensor<Int32>(0))
    let images = _Raw.gatherV2(params: tuple.1, indices: shuffledIndices, axis: Tensor<Int32>(0))
    return (labels, images)
}

public func randomHorizontalFlip<Scalar: TensorFlowFloatingPoint>(images: Tensor<Scalar>,
                                                           horizontalAxis: Int
    ) -> Tensor<Scalar> {
    if Float.random(in: 0 ..< 1) > 0.5 {
        return _Raw.reverseV2(images, axis: Tensor<Int32>([Int32(horizontalAxis)]))
    }
    return images
}

public func makeBatches<T, U>(_ batchSize: Int, _ data: (Tensor<T>, Tensor<U>)) -> [(Tensor<T>, Tensor<U>)] {
  let numExamples = data.0.shape[0]
  let totalBatches = Int(floor(Float(numExamples) / Float(batchSize)))
  var output: [(Tensor<T>, Tensor<U>)] = []
  for ii in 0 ..< totalBatches {
    let startIdx = Int(ii * batchSize)
    let endIdx = Int((ii + 1) * batchSize)
    let labels = data.0
    let images = data.1
    output.append((labels[startIdx ..< endIdx], images[startIdx ..< endIdx]))
  }
  return output
}

let betaDistribution = BetaDistribution(alpha: 0.2, beta: 0.2)
var generator = ThreefryRandomNumberGenerator(seed: [250])

public func mixup(_ batch: (Tensor<Float>, Tensor<Float>)) -> (Tensor<Float>, Tensor<Float>) {
    var (labels1, images1) = batch
    var (labels2, images2) = shuffle(tuple: batch)
    let batchSize = labels1.shape[0]
    
    images2 = randomHorizontalFlip(images: images2, horizontalAxis: 2)
    //create mix ratios and apply them to two arrays
    var scalars: [Float] = []
    for _ in 0 ..< batchSize {
        scalars.append(betaDistribution.next(using: &generator))
    }
    var mixRatio = Tensor<Float>(shape: TensorShape([batchSize, 1]), scalars: scalars)
    mixRatio = max(mixRatio, 1 - mixRatio)
    let labels = labels1 * mixRatio + labels2 * (1 - mixRatio)
    mixRatio = mixRatio.reshaped(to: [batchSize, 1, 1, 1])
    let images = images1 * mixRatio + images2 * (1 - mixRatio)
    //return tuple instead of dataset because creating a new dataset each epoch causes memory issues
    return (labels, images)
    
}
func createOffsets(imageSize: Int32 = 32,
                   glimpseHeight: Int32,
                   glimpseWidth: Int32,
                   batchSize: Int
   ) -> [Tensor<Float>] {
       let pairs = [(glimpseHeight, glimpseWidth),
                    (imageSize - glimpseHeight, glimpseWidth),
                    (glimpseHeight, imageSize - glimpseWidth),
                    (imageSize - glimpseHeight, imageSize - glimpseWidth)]
       return pairs.map { (h, w) -> Tensor<Float> in
            var scalars: [Tensor<Float>] = []
            let borderH = (h + 1) / 2
            let borderW = (w + 1) / 2
            for _ in 0..<batchSize {
                let centerH = Float(Int32.random(in: borderH ... (imageSize - borderH)))
                let centerW = Float(Int32.random(in: borderW ... (imageSize - borderW)))
                scalars.append(Tensor<Float>([centerH, centerW]))
            }
            return Tensor<Float>(stacking: scalars)
       }
}

func createGlimpses(images: [Tensor<Float>], offsets: [Tensor<Float>], sizes: [Tensor<Int32>]) -> [Tensor<Float>] {
    var ret: [Tensor<Float>] = []
    for ii in 0..<images.count {
        ret.append(Tensor<Float>(
            _Raw.extractGlimpse(
                Tensor<Float>(images[ii]),
                size: sizes[ii],
                offsets: offsets[ii],
                centered: false,
                normalized: false,
                uniformNoise: false,
                noise: "zero"
            )))
    }
    return ret
}
func proportion(h: Int32, w: Int32, area: Int32) -> Float {
    return Float(h * w) / Float(area)
}

public func ricap(_ batch: (Tensor<Float>, Tensor<Float>),
                  dataFormat: _Raw.DataFormat = .nhwc) -> (Tensor<Float>, Tensor<Float>) {
    var images1: Tensor<Float>
    if dataFormat == .nchw {
        images1 = batch.1.transposed(permutation: [0, 2, 3, 1])
    } else {
        images1 = batch.1
    }
    
    var labels1 = batch.0
    var (labels2, images2) = shuffle(tuple: (labels1, images1))
    var (labels3, images3) = shuffle(tuple: (labels1, images1))
    var (labels4, images4) = shuffle(tuple: (labels1, images1))
    images1 = randomHorizontalFlip(images: images1, horizontalAxis: 2)
    images2 = randomHorizontalFlip(images: images2, horizontalAxis: 2)
    images3 = randomHorizontalFlip(images: images3, horizontalAxis: 2)
    images4 = randomHorizontalFlip(images: images4, horizontalAxis: 2)
    let batchSize = labels1.shape[0]
    let max: Int32 = 21
    let imageSize: Int32 = 32
    let area = imageSize * imageSize
    // coordinates in a single image that determine the size of the four images pieces
    let h = Int32(round(betaDistribution.next(using: &generator) * Float(imageSize)))
    let w = Int32(round(betaDistribution.next(using: &generator) * Float(imageSize)))
    let offsets = createOffsets(glimpseHeight: h, glimpseWidth: w, batchSize: batchSize)
    let glimpses = createGlimpses( images: [images1, images2, images3, images4],
                                  offsets: offsets,
                                  sizes: [Tensor<Int32>([h, w]),
                                          Tensor<Int32>([imageSize - h, w]),
                                          Tensor<Int32>([h, imageSize - w]),
                                          Tensor<Int32>([imageSize - h, imageSize - w])])
    
    let rightImages = glimpses[0].concatenated(with: glimpses[1], alongAxis: 1)
    let leftImages = glimpses[2].concatenated(with: glimpses[3], alongAxis: 1)
    
    let newLabels = labels1 * proportion(h: h, w: w, area: area) +
                    labels2 * proportion(h: imageSize - h, w: w, area: area) +
                    labels3 * proportion(h: h, w: imageSize - w, area: area) +
                    labels4 * proportion(h: imageSize - h, w: imageSize - w, area: area)
    let result = (newLabels, leftImages.concatenated(with: rightImages, alongAxis: 2))
    if dataFormat == .nhwc{
        return result
    }
    return (result.0, result.1.transposed(permutation: [0, 3, 1, 2]))
}

public func jitter<Scalar: TensorFlowFloatingPoint>(_ nchwInput: Tensor<Scalar>) -> Tensor<Scalar>{
    let input = nchwInput.transposed(permutation: [0,2,3,1])
    let padSize = Int(4)
    let paddedInput = input.padded(forSizes: [(before: 0, after: 0),
                                              (before: padSize, after: padSize),
                                              (before: padSize, after: padSize),
                                              (before: 0, after: 0)],
                                    mode: .reflect)
    let shape = paddedInput.shape
    let batchSize = shape[0]
    let center = Int32(shape[2]) / 2
    var scalars: [Float] = []
    for _ in 0..<batchSize * 2 {
        scalars.append(Float(Int32.random(in: (-Int32(padSize)) ... Int32(padSize))))
    }
    let offsets = Tensor<Float>(shape: [batchSize,2], scalars: scalars)
    return Tensor<Scalar>(
        _Raw.extractGlimpse(
            Tensor<Float>(paddedInput),
            size: input.shapeTensor[1...2],
            offsets: offsets,
            centered: true,
            normalized: false,
            uniformNoise: false,
            noise: "zero"
        ).transposed(permutation: [0, 3, 1, 2])
   )
}
