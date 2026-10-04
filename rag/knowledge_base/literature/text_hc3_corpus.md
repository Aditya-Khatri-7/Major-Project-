---
modality: text
---
# HC3: human versus ChatGPT answers

The Human ChatGPT Comparison Corpus (Guo et al., 2023) pairs thousands of questions with both human
answers and ChatGPT answers across domains such as open QA, finance, medicine, Reddit ELI5 and
Wikipedia-style questions. The authors observed that ChatGPT answers tend to be more formal, more
structured, more neutral in tone and more evenly organised than human answers.

HC3 is convenient for training a first classifier, but it covers essentially one generator family
and one question-answer style. A classifier trained only on HC3 can learn ChatGPT's style rather than
a general notion of machine text, so it must be tested on other generators and domains before its
score is trusted. Splits should be made by question so the same question never appears in both
training and test data.
