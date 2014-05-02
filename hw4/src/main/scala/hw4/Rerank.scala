package hw4

import breeze.linalg.DenseVector
import java.io.File
import RichFile.enrichFile
import breeze.optimize.{StochasticGradientDescent, StochasticDiffFunction, BatchDiffFunction, AdaptiveGradientDescent}

/**
 * Contains a map from label to index, and from index to label
 * @param labels the set of labels to index
 * @tparam T the type of each label
 */
class Alphabet[T](private val labels: TraversableOnce[T]) {
  /** map from index to label */
  lazy val labelOf: Vector[T] = labels.toSet.toVector
  /** map from label to index */
  lazy val indexOf: Map[T, Int] = Map() ++ labelOf.zipWithIndex
  /** the number of unique labels **/
  lazy val size: Int = labelOf.size

  override def toString = labelOf.mkString(", ")
}

trait FeatureExtractor[T] {
  def vocab: Alphabet[String]

  def features(input: T): TraversableOnce[(String, Double)]

  def values(input: T): DenseVector[Double] = {
    val vs = DenseVector.zeros[Double](vocab.size)
    features(input).foreach({ case (k, v) => vs(vocab.indexOf(k)) = v})
    vs
  }
}

case class HypothesisFeatureExtractor(vocab: Alphabet[String]) extends FeatureExtractor[Hypothesis] {
  def features(input: Hypothesis): TraversableOnce[(String, Double)] = {
    input.features ++ Seq(
      "numNonAsciiWords" -> input.words.count(_.exists(127 <)).toDouble,
      "length" -> input.words.size.toDouble,
      "numCommas" -> input.words.count("," ==).toDouble
    )
  }
}

trait Model[-T] {
  def score(input: T): Double
}

case class Hypothesis(words: Array[String], features: Array[(String, Double)])

case class RerankerModel(fe: FeatureExtractor[Hypothesis], params: DenseVector[Double]) extends Model[Hypothesis] {
  val size = params.size

  def values(input: Hypothesis): DenseVector[Double] = {
    val vs = DenseVector.zeros[Double](fe.vocab.size)
    fe.features(input).foreach({ case (k, v) => vs(fe.vocab.indexOf(k)) = v})
    vs
  }

  override def score(input: Hypothesis): Double = values(input) dot params

  def runoff(hypotheses: Seq[Hypothesis]) = {
    val featureValues = hypotheses.map(values)
    val numBeating = featureValues.map { a =>
      featureValues.map(b => params dot (a - b)).count(0 <)
    }
    (hypotheses zip numBeating).maxBy(_._2)._1
  }

  def save(file: File) {
    val lines = for ((name, value) <- fe.vocab.labelOf zip params.toArray) yield {
      "%s\t%s".format(name, value)
    }
    file.writeLines(lines)
  }
}
object RerankerModel {
  val HYPOTHESES_PER_SENTENCE = 100
  lazy val defaultVocab = new Alphabet(Set("p(e)", "p(e|f)", "p_lex(f|e)", "numNonAsciiWords", "length", "numCommas"))
  //  lazy val default = RerankerModel(defaultVocab, HypothesisFeatureExtractor, DenseVector(1f, .5f, .5f))
  private val adaGrad = new AdaptiveGradientDescent.L2Regularization[DenseVector[Double]](1.0, 1.0, 100, 1e-8, 1e-8)
  //  private val sgd = StochasticGradientDescent[DenseVector[Double]]()

  def loadHypotheses(file: File): Iterator[Seq[Hypothesis]] = {
    val lines = file.lines filter { _.nonEmpty } map { line =>
      val Array(_, hypothesis, featureStr) = line.split(""" \|\|\| """)
      val features = featureStr.split(" ") map { col =>
        val Array(feature, value) =  col.split("=")
        feature -> value.toDouble
      }
      Hypothesis(hypothesis.split(" "), features)
    }
    lines.grouped(HYPOTHESES_PER_SENTENCE)
  }

  def loadScores(file: File): Iterator[Seq[Double]] = file.lines.map(_.toDouble).grouped(HYPOTHESES_PER_SENTENCE)

  def train(diffFn: StochasticDiffFunction[DenseVector[Double]]): DenseVector[Double] = {
    adaGrad.minimize(diffFn, DenseVector.zeros[Double](defaultVocab.size))
    //    sgd.minimize(diffFn, DenseVector.zeros[Double](defaultVocab.size))
  }

  def load(file: File): RerankerModel = {
    val namesAndWeights = file.lines.map(line => {
      val Array(name, weight) = line.split("\t")
      (name, weight.toDouble)
    }).toSeq
    val vocab = new Alphabet(namesAndWeights.map(_._1))
    val params = DenseVector.zeros[Double](vocab.size)
    for ((name, weight) <- namesAndWeights) {
      params(vocab.indexOf(name)) = weight
    }
    println(vocab.labelOf, params)
    RerankerModel(HypothesisFeatureExtractor(vocab), params)
  }
}


case class RerankerDiffFunction(vocab: Alphabet[String], data: IndexedSeq[Seq[(Hypothesis, Double)]])
    extends BatchDiffFunction[DenseVector[Double]] {

  import RerankerModel.HYPOTHESES_PER_SENTENCE

  private val featureExtractor = HypothesisFeatureExtractor(vocab)
  private val featureValues = data map (_ map {case (h, s) => (featureExtractor.values(h), s) })
  private val numPairsPerSentence = HYPOTHESES_PER_SENTENCE * HYPOTHESES_PER_SENTENCE

  private def sumPair(a: (Double, DenseVector[Double]), b: (Double, DenseVector[Double])) = (a._1 + b._1, a._2 + b._2)
  private def sign(x: Double): Double = if (x > 0) 1.0 else if (x < 0) -1.0 else 0.0
  private def sign(v: DenseVector[Double]): DenseVector[Double] = v.map(sign)

  override val fullRange = 0 until numPairsPerSentence * data.size

  /** 1-1 map from (0 until numPairsPerSentence * data.size) onto pairs of hypotheses */
  private def pairFromIdx(i: Int) = {
    val sentId = i / numPairsPerSentence
    val remainder = i % numPairsPerSentence
    val a = remainder / HYPOTHESES_PER_SENTENCE
    val b = remainder % HYPOTHESES_PER_SENTENCE
    val sentHyps = featureValues(sentId)
    (sentHyps(a), sentHyps(b))
  }

  override def calculate(x: DenseVector[Double], batch: IndexedSeq[Int]): (Double, DenseVector[Double]) = {
    val ((obj, grad), elapsed) = Timer.time {
      val m = RerankerModel(featureExtractor, x)
      val objsAndGrads = for (
        ((a, aScore), (b, bScore)) <- batch.par.map(pairFromIdx);
        diff = a - b;
        predicted = sign(m.params dot diff);
        correct = sign(aScore - bScore)
        if predicted != correct // wrong prediction
      ) yield {
        // perceptron loss, gradient
        val cost = 1.0 // if (math.abs(aScore - bScore) < 0.05) .5 else 1.0 // partial cost
        val grad = sign(diff) * (-correct * cost)
        (cost, grad)
      }
      objsAndGrads.reduce(sumPair)
    }
    println("objective: %s   gradnorm: %s   millis elapsed: %s".format(obj, grad.norm(2), elapsed / 1000))
    (obj, grad)
  }
}

object Train extends App {
  import RerankerModel._

  val batchSize = 1000000
  val hypothesisFile = new File("data/dev.100best")
  val scoresFile = new File("data/dev.all.scores")
  val modelFile = new File("model.tsv")

  val data = for ((hs, ss) <- loadHypotheses(hypothesisFile) zip loadScores(scoresFile)) yield { hs zip ss }
  val traineddParams = train(RerankerDiffFunction(defaultVocab, data.toVector).withRandomBatches(batchSize))
  println(defaultVocab)
  println(traineddParams)
  RerankerModel(HypothesisFeatureExtractor(defaultVocab), traineddParams).save(modelFile)
}

object Rerank extends App {
  import RerankerModel._

  val hypothesisFile = new File("data/test.100best")
  val outputFile = new File("test.reranked")
  val modelFile = new File("model.tsv")

  val model = RerankerModel.load(modelFile)
  val hypotheses = loadHypotheses(hypothesisFile)
//  outputFile.writeLines(hypotheses map { _.head.words.mkString(" ") })
  outputFile.writeLines(hypotheses map { h => model.runoff(h).words.mkString(" ") })
}
