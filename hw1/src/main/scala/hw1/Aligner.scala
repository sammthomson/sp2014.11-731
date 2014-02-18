package hw1

import scala.io.{Codec, Source}
import scala.collection.mutable
import math.{log, exp}
import java.io.File
import resource.managed
import RichFile.enrichFile
import scala.collection.parallel.ParIterable
import LogProbRing._


case class SentencePair(srcSentence: Seq[String], targetSentence: Seq[String])

case class Counts(var counts: mutable.Map[String, Float], var z: Float) {
  import Counts._

  def += (other: Counts): Counts = {
    z = plus(z, other.z)
    counts = merge(counts, other.counts)
    this
  }
}
object Counts {
  def empty = Counts(mutable.Map(), one)

  def merge[K](left: mutable.Map[K, Float], right: Iterable[(K, Float)]): mutable.Map[K, Float] = {
    left ++= right map {
      case (k, v) => k -> plus(left.getOrElse(k, zero), v)
    }
  }

  def mergeNested[K, L](left: mutable.Map[K, mutable.Map[L, Float]],
                        right: Iterable[(K, mutable.Map[L, Float])]): mutable.Map[K, mutable.Map[L, Float]] = {
    left ++= right map {
      case (k, v) => k -> merge(left.getOrElse(k, mutable.Map()), v)
    }
  }

  def mergeMapAndLL(left: (mutable.Map[String, mutable.Map[String, Float]], Float),
                    right: (Iterable[(String, mutable.Map[String, Float])], Float)): (mutable.Map[String, mutable.Map[String, Float]], Float) = {
    (mergeNested(left._1, right._1), times(left._2, right._2))
  }
}

case class Model1(alignmentModel: AlignmentModel, translationModel: WordTranslationModel) {
  import Counts._

  val NULL = "**NULL**"
  val MIN_LOG_PROB = -5f
//  val MAX_ALIGNMENT_DISTANCE = 15

  def takeEMStep(trainingData: Traversable[SentencePair]): Model1 = {
    System.err.println("\ngetting expected counts")
    val counts = getAllExpectedCounts(trainingData)
    // turn into P( target | src )
    System.err.println("\nnormalizing to get P(target | src)")
    val probs = counts.zipWithIndex.map({ case ((srcToken, targetProbs), i) => {
      if (i % 100 == 0) {
        System.err.print(".")
        System.err.flush()
      }
      val srcProb = sum(targetProbs.values)
      srcToken -> targetProbs.map({
        case (targetToken, prob) => targetToken -> dividedBy(prob, srcProb)
      }).filter(_._2 > MIN_LOG_PROB)
    }}).filterNot(_._2.isEmpty)
    System.err.println("\ndone normalizing")
    Model1(alignmentModel, PhraseTable(probs, zero))
  }

  def getAllExpectedCounts(trainingData: Traversable[SentencePair]): mutable.Map[String, mutable.Map[String, Float]] = {
    val allCounts = trainingData.toIterator.grouped(100).map {
      System.err.print(",")
      System.err.flush()
      _.par.map(getExpectedCounts).reduce(mergeMapAndLL)
    }
    val (results, totalLogLikelihood) = allCounts.reduceLeft(mergeMapAndLL)
    System.err.println("\nLog Likelihood: %s".format(totalLogLikelihood))
    results
  }

  def getExpectedCounts(pair: SentencePair): (mutable.Map[String, mutable.Map[String, Float]], Float) = {
    val SentencePair(srcSentence, targetSentence) = pair
    val m = srcSentence.length
    val n = targetSentence.length
    val countsAndLogLiks = targetSentence.zipWithIndex.view.par.map({ case (targetToken, j) => {
        val scores = srcSentence.zipWithIndex.map {
          case (srcToken, i) => times(translationModel.score(srcToken, targetToken), alignmentModel.score(i, j, m, n))
        } :+ translationModel.score(NULL, targetToken)
        val targetTokenTotal = sum(scores)
        val normalized = scores.map(dividedBy(_, targetTokenTotal))
        val contributions = (srcSentence :+ NULL zip normalized).filter(_._2 > MIN_LOG_PROB).map {
          case (srcToken, score) => srcToken -> mutable.Map(targetToken -> score)
        }
        (contributions, targetTokenTotal)
      }
    }).seq
    countsAndLogLiks.foldLeft((mutable.Map[String, mutable.Map[String, Float]](), one))(mergeMapAndLL)
  }

  def decode(pair: SentencePair, offDiagPenalty: Float = 2f): Seq[(Int, Int)] = {
    val m = pair.srcSentence.length
    val n = pair.targetSentence.length
    for ((targetToken, j) <- pair.targetSentence.zipWithIndex;
         bestAlignment = pair.srcSentence.zipWithIndex.map { case (src, i) =>
           times(translationModel.score(src, targetToken), alignmentModel.score(i, j, m, n))
         }.zipWithIndex.maxBy(_._1)
         if bestAlignment._1 > translationModel.score(NULL, targetToken)) yield {
      (bestAlignment._2, j)
    }
  }

  def decodeAll(sentences: TraversableOnce[SentencePair], offDiagPenalty: Float = 2f): Iterator[String] = {
    sentences.toIterator.grouped(100).flatMap( group => {
      System.err.print(".")
      System.err.flush()
      group.par.map(decode(_, offDiagPenalty)).map(_.map({ case (i, j) => "%d-%d".format(i, j) }).mkString(" ")).seq
    })
  }

  def save(file: File) = translationModel.save(file)
}

object Model1 {
  val PAIR_SEPARATOR = """ \|\|\| """

  val flat = Model1(UniformAlignmentModel, PhraseTable(Map(), one))

  def readSentencePairs(bitextFilename: String, numSentences: Int): Array[SentencePair] = {
    System.err.println("reading sentences")
    var pairs: Array[SentencePair] = null
    for(bitextFile <- managed(Source.fromFile(bitextFilename)(Codec.UTF8))) {
      pairs = (bitextFile.getLines().take(numSentences).map {
        line => line.split(PAIR_SEPARATOR).map(_.trim().split(" ").toSeq)
      } map {
        case Array(src, target) => SentencePair(src, target)
      }).toArray
    }
    System.err.println("done")
    pairs
  }
}

object LogProbRing {
  val zero = Float.NegativeInfinity
  val one = 0.0f

  def plus(a: Float, b: Float): Float = log(exp(a) + exp(b)).toFloat

  def times(a: Float, b: Float): Float = a + b

  def minus(a: Float, b: Float): Float = log(exp(a) - exp(b)).toFloat

  def dividedBy(a: Float, b: Float): Float = a - b

  def sum(xs: Traversable[Float]): Float = if (xs.isEmpty) zero else sum(xs.par)
  def sum(xs: ParIterable[Float]): Float = {
    val m = xs.max
    (m + log(xs.map(d => exp(d - m)).sum)).toFloat
  }

  def product(xs: Traversable[Float]): Float = xs.par.sum
}

object Settings {
  val bitextFilename = "/Users/sam/code/sp2014.11-731/hw1/data/dev-test-train.de-en"
  val experimentsDir = "experiments"
}

object RunEM extends App {
  import Settings._

  val numSentences = args(0).toInt
  val outputFolder = new File(experimentsDir, args(1))
  val offDiagPenalty = args(2).toFloat
  var model = Model1(DiagonalAlignmentModel(offDiagPenalty), PhraseTable(Map(), one))
  val numIterations = 10
  val sentences = Model1.readSentencePairs(bitextFilename, numSentences)
  outputFolder.mkdir()
  val outputPredictionsFile = new File(outputFolder, "predictions.txt")
  (1 to numIterations) foreach { i =>
    model = model.takeEMStep(sentences)
    val modelFile = new File(outputFolder, "model_%s.txt".format(i))
    System.err.println("saving model to %s".format(modelFile.getCanonicalPath))
    model.save(modelFile)
    System.err.println("done.")
  }
  private val predictions = model.decodeAll(sentences.iterator, offDiagPenalty)
  outputPredictionsFile.writeLines(predictions)
}


object Predict extends App {
  import Settings._

  val numSentences = args(0).toInt
  val outputPredictionsFile = new File(args(1))
  val modelFile = new File(args(2))
  val offDiagPenalty = args(3).toFloat
  var model = Model1(DiagonalAlignmentModel(offDiagPenalty), PhraseTable.load(modelFile))
  val sentences = Model1.readSentencePairs(bitextFilename, numSentences)
  private val predictions = model.decodeAll(sentences.iterator, offDiagPenalty)
  outputPredictionsFile.writeLines(predictions)
}
