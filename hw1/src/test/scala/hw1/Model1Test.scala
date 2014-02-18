package hw1

import org.scalatest.FlatSpec
import org.scalatest.matchers.ShouldMatchers


class Model1Test extends FlatSpec with ShouldMatchers {
  "A Model1" should "get expected counts right" in {
    val model = Model1.flat
    val sentence = Model1.readSentencePairs(Settings.bitextFilename, 1).head
    val counts = model.getExpectedCounts(sentence.srcSentence, sentence.targetSentence)
    counts._1.foreach(println)
  }
}
