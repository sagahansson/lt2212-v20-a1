# LT2212 V20 Assignment 1

Put any documentation here including any answers to the questions in the 
assignment on Canvas.

Part 1 -- word/term choices
I chose to only count alphabetic characters as words, i.e. numbers and punctuation was removed. The reason behind this is that I don't think numbers and punctuation add the same amount of meaning as a word does (although it obviously adds some). I also chose to lowercase all words, since, for this assignment, it seemed like there was no purpose in keeping lowercased and uppercased/word-initial uppercased words separate. That is, for this task, I did not need to know hom many "the" are part of a header ("THE") or the beginning of a sentence ("The").

Part 4 -- changes in visualising
The changes between visualising part1 and part3 are substantial -- common words are sorted out, meaning that more meaningful words for each class are shown. That is, when visualising part1, words such as "the", "to", and "of" are among the most frequent. When visualising part3, on the other hand, words such as "tonnes", "grain" and "oil" are among the most frequent. "Tonnes" and "grain" are most frequent in the grain class, whereas "oil" is most frequent in the crude class. A few words, such as "mln" and "the", that are almost equally frequent in both classes do appear in this table as well, but not nearly as many as when visualising part1.

Part Bonus -- classifying scores
The accuracy scores I got were 0.8922413793103449 for classifying with the dataframe from part1, and 0.9482758620689655 for classifying with the dataframe from part3, i.e. the accuracy score is higher when using tf-idf. This is expected, as tf-idf is meant to highlight words of importance for each class (which it does, as the visualising clearly demonstrates). 