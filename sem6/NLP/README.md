1. What is NLTK?

- Definition: NLTK stands for Natural Language Toolkit. It is a comprehensive and widely used open-source Python library designed for research and development in Natural Language Processing (NLP) and related fields like computational linguistics, cognitive science, and artificial intelligence.
- Purpose: NLTK provides a vast collection of tools, algorithms, and resources to perform various NLP tasks. It's often used for learning, teaching, prototyping, and research in NLP.
- Key Features:
  - Text Processing: Tokenization (splitting text into words/sentences), stemming (reducing words to root form), lemmatization (reducing words to base/dictionary form).
  - Syntactic Analysis: Part-of-Speech (POS) tagging (assigning grammatical categories like noun, verb), parsing (analyzing sentence structure).
  - Semantic Analysis: Access to lexical resources like WordNet for exploring word meanings and relationships.
  - Corpora and Lexical Resources: Includes interfaces to numerous corpora (large text collections) and lexical databases (like WordNet).
  - Classification: Tools for text classification tasks (e.g., sentiment analysis).
- Example: Your lab manual uses NLTK for POS tagging (Experiment 3) and accessing WordNet (Experiment 4). A simple NLTK usage example is `nltk.word_tokenize("This is an example.")` which would output `['This', 'is', 'an', 'example', '.']`.

---

2. What is VSM, vector space model?

- Definition: The Vector Space Model (VSM) is an algebraic model used for representing text documents (or any objects) as vectors of identifiers, such as index terms (words).
- How it works:
  1.  A vocabulary of all unique terms across a collection of documents is created.
  2.  Each term in the vocabulary corresponds to a dimension in a high-dimensional space.
  3.  Each document is represented as a vector in this space.
  4.  The value (component) of the vector along each dimension typically represents the importance or frequency of the corresponding term in that document. Common weighting schemes include:
      - Term Frequency (TF): How often a term appears in a document.
      - TF-IDF (Term Frequency-Inverse Document Frequency): Weights terms by how often they appear in a document (TF) but penalizes terms that appear in many documents (IDF), highlighting terms that are more specific to a particular document. (Mentioned as a comparison point in Experiment 8/N-Gram description).
- Purpose: VSM is primarily used in information retrieval, document similarity calculation, and text classification. By representing documents as vectors, we can calculate the similarity between them using measures like cosine similarity (measuring the angle between vectors). Documents with similar content will have vectors pointing in similar directions (smaller angle, higher cosine similarity).
- Example:
  - Doc 1: "The cat sat on the mat."
  - Doc 2: "The dog sat on the log."
  - Simplified Vocabulary (ignoring stop words): {cat, sat, mat, dog, log}
  - Vector Doc 1 (TF): [1, 1, 1, 0, 0]
  - Vector Doc 2 (TF): [0, 1, 0, 1, 1]
    We could then calculate the cosine similarity between these vectors.

---

3. What is POS? What is POS tagging?

- POS (Part-of-Speech): Refers to the grammatical category or class of a word based on its syntactic function in a sentence. Common POS categories include Noun (NN), Verb (VB), Adjective (JJ), Adverb (RB), Preposition (IN), Determiner (DT), Pronoun (PRP), etc. The specific tags (like NN, VBZ, JJ) often come from standard tagsets like the Penn Treebank tagset.
- POS Tagging: This is the _process_ of assigning the appropriate Part-of-Speech tag to each word (token) in a given text or sentence. The tagging is context-dependent, meaning the same word can have different POS tags depending on how it's used (e.g., "book" can be a noun or a verb).
- Purpose: POS tagging is a fundamental step in many NLP pipelines. It helps in understanding the grammatical structure of a sentence, which is crucial for tasks like:
  - Syntactic Parsing (analyzing sentence structure)
  - Named Entity Recognition (identifying proper nouns)
  - Information Extraction
  - Machine Translation
  - Disambiguation (e.g., distinguishing "conduct" as a noun vs. verb)
- Example: As shown in Experiment 3:
  - Sentence: "The quick brown fox jumps over the lazy dog."
  - POS Tagged: `[('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'), ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN'), ('.', '.')]`

---

4. What is semantic analysis?

- Definition: Semantic analysis is a stage in NLP that focuses on understanding the _meaning_ of words, phrases, sentences, and larger texts. It goes beyond the grammatical structure (syntax) to interpret the intended meaning and logical relationships.
- Focus:
  - Lexical Semantics: Meaning of individual words.
  - Word Sense Disambiguation (WSD): Determining the correct meaning of a word when it has multiple senses (e.g., "bank" as a financial institution vs. a river bank).
  - Semantic Role Labeling (SRL): Identifying the roles (like Agent, Patient, Instrument) that words play in relation to a predicate (typically a verb). E.g., in "John broke the window with a hammer", John=Agent, window=Patient, hammer=Instrument. (Related to PropBank).
  - Relationship Extraction: Identifying semantic relationships between entities (e.g., "Located In", "Employed By").
  - Understanding synonyms, antonyms, hyponyms, hypernyms (as explored in Experiment 4 using WordNet).
- Goal: To represent the meaning of the text in a structured way that a computer can process and reason with.
- Example: Understanding that "The dog chased the cat" and "The cat was chased by the dog" describe the same essential event, despite different syntactic structures. Identifying that "buy" and "purchase" are synonyms in many contexts.

---

5. What is syntactic analysis?

- Definition: Syntactic analysis, also known as parsing, is the process of analyzing the grammatical structure of a sentence according to the rules of a formal grammar. It determines how words group together to form phrases (like Noun Phrases, Verb Phrases) and how these phrases form sentences.
- Focus: Identifying the hierarchical structure and dependencies between words in a sentence. It checks if a sentence is grammatically correct and reveals its underlying organization.
- Output: Often represented as a parse tree (either a constituency tree showing phrase structures or a dependency tree showing head-word and dependent relationships).
- Relationship to POS Tagging: POS tagging is usually a prerequisite for syntactic analysis. The parser uses the POS tags to help determine the sentence structure.
- Example: For the sentence "The cat chased the mouse", a constituency parse might look like:
  `(S (NP (DT The) (NN cat)) (VP (VBD chased) (NP (DT the) (NN mouse))))`
  This shows the sentence (S) is composed of a Noun Phrase (NP) and a Verb Phrase (VP), which are further broken down.

---

6. What is sentimental analysis?

- Definition: Sentiment analysis (or opinion mining) is the use of NLP, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. Essentially, it aims to determine the emotional tone or attitude expressed in a piece of text.
- Focus:
  - Polarity Classification: Categorizing text as positive, negative, or neutral.
  - Emotion Detection: Identifying specific emotions like happiness, sadness, anger, fear.
  - Aspect-Based Sentiment Analysis (ABSA): Identifying opinions about specific aspects or features of an entity (e.g., "The phone's screen is great [positive], but the battery life is terrible [negative]").
  - Subjectivity Detection: Distinguishing factual text from opinionated text.
- Applications: Widely used for analyzing customer reviews, social media monitoring, brand perception, market research, political polling, etc. (Mentioned as an application of N-Grams in Experiment 8).
- Example:
  - "I loved the movie, it was fantastic!" -> Positive
  - "The flight was delayed and the service was poor." -> Negative
  - "The meeting is scheduled for 3 PM." -> Neutral

---

7. What is the wordnet? Why is it called a lexical database?

- WordNet: WordNet is a large electronic lexical database of English. It was developed at Princeton University.
- Structure & Content: It groups English words (nouns, verbs, adjectives, adverbs) into sets of cognitive synonyms called synsets. Each synset represents a distinct concept or meaning. Synsets are interlinked using conceptual-semantic and lexical relations.
- Key Features:
  - Provides short definitions (glosses) and example sentences for many synsets.
  - Organizes words based on meaning, not just form.
  - Defines various semantic relationships between synsets, including:
    _ Synonymy: Words in the same synset (e.g., {car, auto, automobile}).
    _ Antonymy: Opposite meanings (e.g., good vs. bad).
    _ Hyponymy/Hypernymy: Type-of relationships (e.g., "dog" is a hyponym of "animal"; "animal" is a hypernym of "dog").
    _ Meronymy/Holonymy: Part-of relationships (e.g., "wheel" is a meronym of "car"; "car" is a holonym of "wheel").
    (These relationships are explored in Experiment 4).
- Why "Lexical Database": It's called a lexical database because:
  - Lexical: It deals with words (lexemes) and their properties (meaning, relationships).
  - Database: It's a large, structured, organized collection of this lexical information, designed for computational access and querying, unlike a traditional paper dictionary. It stores not just definitions but also the intricate network of relationships between word meanings.

---

8. What is the difference between the word net and verb net?

- Focus:
  - WordNet: Covers nouns, verbs, adjectives, and adverbs. Focuses broadly on semantic relationships like synonymy, hypernymy, meronymy, etc.
  - VerbNet: Focuses _exclusively_ on verbs and their syntactic and semantic properties.
- Organization:
  - WordNet: Organized around synsets (sets of synonyms) linked by various semantic relations.
  - VerbNet: Organized into verb classes. Verbs within a class share common syntactic behavior (how they structure sentences, e.g., transitive, intransitive) and semantic roles (the roles their arguments play, like Agent, Theme, Patient). It explicitly maps syntactic frames to semantic representations.
- Information Provided:
  - WordNet: Provides definitions, synonyms, hypernyms, hyponyms, meronyms, etc.
  - VerbNet: Provides detailed information about the arguments a verb takes (thematic roles), selectional restrictions on those arguments (e.g., the Agent must be animate), and the specific syntactic frames (sentence structures) the verb can appear in.
- Relationship: They are complementary resources. WordNet gives broad lexical coverage and general semantic links, while VerbNet provides deep, structured information specifically about verb behavior. VerbNet often links its senses to WordNet synsets.

---

9. What is propBank what is treeBank?

- TreeBank:
  - Definition: A TreeBank is a corpus (text collection) where each sentence has been annotated with its syntactic structure, usually in the form of a tree (either a constituency tree or a dependency tree). It typically includes Part-of-Speech tagging as well.
  - Purpose: TreeBanks are essential resources for training and evaluating syntactic parsers, studying linguistic phenomena, and providing a structural foundation for further semantic annotation.
  - Example: The Penn Treebank, containing parsed articles from the Wall Street Journal, is the most famous example.
- PropBank (Proposition Bank):
  - Definition: PropBank is a corpus annotated with semantic roles or predicate-argument structures, primarily for verbs. It adds a layer of shallow semantic information ("who did what to whom") onto the syntactic structures provided by a TreeBank (often the Penn TreeBank).
  - Annotation: For each verb (predicate) in a sentence, PropBank identifies its arguments and assigns them numbered roles like `Arg0` (typically the Agent or Causer), `Arg1` (Patient or Theme), `Arg2` (Instrument, Beneficiary, etc.), as well as modifier roles like `ArgM-LOC` (Location), `ArgM-TMP` (Time). The specific meaning of `Arg0`, `Arg1`, etc., is defined individually for each verb sense.
  - Purpose: Used for training and evaluating semantic role labeling (SRL) systems, which are important for understanding event structures in text.
- Relationship: PropBank annotation is typically performed _on top of_ a TreeBank. The TreeBank provides the syntactic structure, and PropBank adds the semantic argument labels to the constituents identified in that structure.

---

10. What is semantic machine translation?

- Definition: Semantic Machine Translation (SMT) refers to approaches in Machine Translation (MT) that prioritize translating the _meaning_ or _semantics_ of the source text, rather than relying solely on word-for-word substitution or purely syntactic transformations.
- Goal: To produce translations that are not only grammatically correct in the target language but also accurately convey the intended meaning, nuances, and context of the source text. This involves deeper understanding than just surface-level structure.
- How it Differs from Other Approaches:
  - Rule-Based MT (RBMT): Relies on manually crafted linguistic rules (lexical, syntactic, some semantic). Can be brittle.
  - Statistical MT (SMT): Learns translation patterns (phrase mappings) from large parallel corpora. Doesn't explicitly model deep meaning.
  - Neural MT (NMT): Uses deep learning (e.g., sequence-to-sequence models) to learn an end-to-end mapping. Modern NMT models often implicitly capture a significant amount of semantics due to their architecture (like attention mechanisms) but don't necessarily use an explicit intermediate semantic representation.
- Challenges: Representing meaning in a language-independent way is extremely difficult. Early attempts at explicit semantic MT struggled with creating robust meaning representations. Modern NMT often achieves meaning-preserving translation more effectively through implicit learning.
- Example: Accurately translating idioms (e.g., "He kicked the bucket" -> "Il est mort" in French, not a literal translation) requires understanding the semantic meaning. Handling ambiguity based on context is also a key aspect.

---

11. What are the 5 stages of NLP?

While different sources might slightly vary the names or number, a common conceptual model of the stages involved in deep natural language understanding includes these five:

1.  Lexical Analysis:
    - Goal: Identify and analyze the structure of individual words. Break down the text into basic lexical units (tokens).
    - Tasks: Tokenization (splitting text into words/punctuation), lemmatization/stemming (reducing words to base forms), identifying word boundaries, potentially POS tagging (sometimes placed here or in syntax).
2.  Syntactic Analysis (Parsing):
    - Goal: Analyze the grammatical structure of sentences. Determine how words relate to each other based on grammar rules.
    - Tasks: POS tagging (if not done earlier), constituency parsing (identifying phrases), dependency parsing (identifying head-dependent relationships), checking grammatical validity.
3.  Semantic Analysis:
    - Goal: Determine the literal meaning of words, phrases, and sentences, independent of context.
    - Tasks: Word Sense Disambiguation (WSD), mapping syntactic structures to meaning representations (e.g., predicate-argument structure), identifying semantic roles (SRL).
4.  Discourse Integration:
    - Goal: Understand how sentences relate to each other and how meaning unfolds across a larger text. Consider the context beyond single sentences.
    - Tasks: Anaphora resolution (determining what pronouns like "he", "it", "they" refer to), coreference resolution (identifying all expressions referring to the same entity), analyzing coherence and relationships between sentences (e.g., cause-effect, elaboration).
5.  Pragmatic Analysis:
    - Goal: Understand the intended meaning or purpose of the language use in a specific context. Goes beyond the literal meaning to interpret the speaker's/writer's intention.
    - Tasks: Resolving ambiguity based on real-world knowledge and context, understanding implicatures (what is implied but not explicitly stated), identifying speech acts (e.g., request, command, statement), understanding figurative language (metaphors, sarcasm).

_Note: These stages are conceptual levels of analysis. Modern NLP systems, especially deep learning models, often perform these tasks in a more integrated or end-to-end fashion rather than as strictly separate sequential steps._

---

12. What is Spacy? For what purpose it is used?

- Definition: spaCy is a popular, modern open-source software library for advanced Natural Language Processing (NLP) in Python. It is designed specifically for production use and aims to help build real-world applications that process and "understand" large volumes of text.
- Focus: spaCy emphasizes efficiency, speed, accuracy, and ease of use for common NLP tasks required in production environments. It provides pre-trained statistical models and processing pipelines for many languages.
- Purpose & Key Features: It is used for building NLP pipelines that typically include tasks like:
  - Tokenization: Fast and robust splitting of text.
  - Part-of-Speech (POS) Tagging: Assigning grammatical labels.
  - Dependency Parsing: Analyzing grammatical relationships between words.
  - Lemmatization: Reducing words to their base form.
  - Named Entity Recognition (NER): Identifying and classifying named entities like persons, organizations, locations, dates, etc. (as demonstrated in Experiments 6 and 7).
  - Text Classification: Assigning categories or labels to whole texts.
  - Entity Linking: Disambiguating entities to unique identifiers in a knowledge base.
  - Sentence Boundary Detection (SBD): Identifying sentence limits.
- Use Cases: Building chatbots, information extraction systems, text summarization tools, data analysis pipelines, content analysis, etc. It's often preferred over NLTK when building applications that need to be fast and handle large amounts of data efficiently.

---

13. What are types of morphology?

Morphology is the study of word formation and structure. The two primary types, as mentioned in Experiment 1, are:

1.  Inflectional Morphology:
    - Definition: Deals with changes to a word form to express different grammatical functions or attributes such as tense, number, person, case, gender, mood, or aspect.
    - Effect: It creates different _forms_ of the _same_ word (lexeme) but does not change the word's core meaning or its grammatical category (part of speech).
    - Examples:
      - `run` -> `runs` (3rd person singular present tense)
      - `run` -> `running` (present participle/gerund)
      - `run` -> `ran` (past tense)
      - `cat` -> `cats` (plural number)
      - `happy` -> `happier`, `happiest` (comparative, superlative degree)
2.  Derivational Morphology:
    - Definition: Deals with the formation of new words from existing words (roots or stems), often by adding affixes (prefixes or suffixes).
    - Effect: It typically changes the core meaning of the word or its grammatical category (part of speech).
    - Examples:
      - `happy` (adjective) -> `unhappy` (adjective, changed meaning)
      - `happy` (adjective) -> `happiness` (noun, changed part of speech)
      - `run` (verb) -> `runner` (noun)
      - `kind` (adjective) -> `kindness` (noun)
      - `do` (verb) -> `redo` (verb, changed meaning)

_Other related processes sometimes discussed under morphology include:_

- Compounding: Combining two or more free morphemes (words) to create a new word (e.g., `blackboard`, `website`, `sunflower`).

---

14. What are types of morphemes?

A morpheme is the smallest unit of meaning or grammatical function in a language. The main types, as described in Experiment 1, are:

1.  Free Morphemes:
    - Definition: Morphemes that can stand alone as independent words and have meaning on their own.
    - Examples: `cat`, `run`, `happy`, `book`, `on`, `the`.
2.  Bound Morphemes:
    - Definition: Morphemes that cannot stand alone as words; they must be attached (bound) to another morpheme (typically a free morpheme or another bound root) to convey meaning or grammatical information.
    - Subtypes:
      - Affixes: Bound morphemes added to a base or stem.
        - Prefixes: Added before the base (e.g., `un-` in `unhappy`, `re-` in `redo`).
        - Suffixes: Added after the base (e.g., `-s` in `cats`, `-ed` in `walked`, `-ing` in `running`, `-ness` in `kindness`, `-able` in `readable`).
        - _(Infixes: Inserted within a base - rare in English, e.g., 'abso-freaking-lutely')_
        - _(Circumfixes: Added both before and after - not in English)_
      - Bound Roots: Roots that carry core meaning but cannot stand alone (less common focus in introductory NLP, but exist). Example: `-ject` in `reject`, `inject`, `project`; `-ceive` in `receive`, `perceive`.

- Relationship to Morphology Types: Bound morphemes can be further classified based on their function:
  - Inflectional Morphemes: These are always suffixes in English (e.g., `-s`, `-es`, `-ed`, `-ing`, `-en`, `-er`, `-est`). They serve grammatical functions (tense, number, etc.).
  - Derivational Morphemes: These can be prefixes or suffixes (e.g., `un-`, `re-`, `-ness`, `-ly`, `-able`). They create new words or change the word class.
