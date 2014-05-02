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
  def features(input: T): TraversableOnce[(String, Double)]
}

object HypothesisFeatureExtractor extends FeatureExtractor[Hypothesis] {
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

case class RerankerModel(vocab: Alphabet[String],
                         fe: FeatureExtractor[Hypothesis],
                         params: DenseVector[Double])
    extends Model[Hypothesis] {
  val size = params.size

  def values(input: Hypothesis): DenseVector[Double] = {
    val vs = DenseVector.zeros[Double](vocab.size)
    fe.features(input).foreach({ case (k, v) => vs(vocab.indexOf(k)) = v})
    vs
  }

  override def score(input: Hypothesis): Double = values(input) dot params

  def instantRunoff(hypotheses: Seq[Hypothesis]) = {
    var remaining = hypotheses.toVector
    while (remaining.size > 1) {
      val beating = for (a <- remaining) yield {
        for (
          b <- remaining;
          diff: DenseVector[Double] = values(a) - values(b);
          score = params dot diff
          if score > 0
        ) yield 1
      }
      remaining = (remaining zip beating).sortBy(_._2.size).tail.map(_._1)
    }
    remaining.head
  }
}

case class RerankerDiffFunction(vocab: Alphabet[String], data: IndexedSeq[Seq[(Hypothesis, Double)]])
    extends BatchDiffFunction[DenseVector[Double]] {

  import RerankerModel.HYPOTHESES_PER_SENTENCE

  private[this] val numPairsPerSentence = HYPOTHESES_PER_SENTENCE * HYPOTHESES_PER_SENTENCE

  def sumPair(a: (Double, DenseVector[Double]), b: (Double, DenseVector[Double])) = (a._1 + b._1, a._2 + b._2)

  override def calculate(x: DenseVector[Double], batch: IndexedSeq[Int]): (Double, DenseVector[Double]) = {
    val m = RerankerModel(vocab, HypothesisFeatureExtractor, x)
    val objsAndGrads = for (
        ((a, aScore), (b, bScore)) <- batch.par.map(fromIdx);
        diff = m.values(a) - m.values(b);
        score = m.params dot diff
        if ((score >= 0) && (aScore < bScore)) || (score < 0 && aScore > bScore) // wrong prediction
        ) yield {
      (1.0, diff) // perceptron loss, gradient
    }
    val n = 1.0 / batch.size
    val (obj, grad) = objsAndGrads.reduce(sumPair)
    println("objective: %s   gradnorm: %s".format(obj * n, grad.norm(2) * n))
    (obj, grad)
  }

  /** 1-1 map from (0 until numPairsPerSentence * data.size) onto pairs of hypotheses */
  private def fromIdx(i: Int) = {
    val sentId = i / numPairsPerSentence
    val remainder = i % numPairsPerSentence
    val a = remainder / HYPOTHESES_PER_SENTENCE
    val b = remainder % HYPOTHESES_PER_SENTENCE
    val sentHyps = data(sentId)
    (sentHyps(a), sentHyps(b))
  }

  override val fullRange = 0 until numPairsPerSentence * data.size
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
}

object Train extends App {
  import RerankerModel._

  val hypothesisFile = new File("data/dev.100best")
  val scoresFile = new File("data/dev.all.scores")

  val data = for ((hs, ss) <- loadHypotheses(hypothesisFile) zip loadScores(scoresFile)) yield { hs zip ss }

  val newModel = train(RerankerDiffFunction(defaultVocab, data.toVector).withRandomBatches(1000000))
  println(defaultVocab)
  println(newModel)
}

object Rerank extends App {
  import RerankerModel._
  val model = RerankerModel(
    defaultVocab,
    HypothesisFeatureExtractor,
    // p(e|f), numNonAsciiWords, p_lex(f|e), length, numCommas, p(e)
    DenseVector(
      1.357322141154833,
      -0.10255545243388073,
      -2.1378364025552887,
      0.12915390814841818,
      0.8853725580653389,
      0.7157717221907849
    )
  )
//  val hypothesisFile = new File("data/dev.100best")
//  val outputFile = new File("dev.reranked")
  val hypothesisFile = new File("data/test.100best")
  val outputFile = new File("test.reranked")
  val hypotheses = loadHypotheses(hypothesisFile)
  outputFile.writeLines(hypotheses map { h => model.instantRunoff(h).words.mkString(" ") })
}
