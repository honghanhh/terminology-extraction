# Terminology Extraction - Paper Summary

## ATE: Automatic term extraction (TermEval 2020)
<!-- 
- Automatic terminology extraction:
    - Rule-based approaches
        - Conducted by a terminologist
        - Hand-operated exploration, indexation, 
        - Costly maintenance of domain-specific corpora and terminologies 
    - __Feature-based approaches__
    - __Context-based approaches__
    - Hybrid approaches. -->

---

0. Definitions
    - __TermEval 2020__: a platform for researchers to work on ATE.
    - __ATE__: the automated process of identifying terminology from a corpus
    of specialised texts.
    - __Terms__: lexical items that represent concepts of a domain.

1.  [Shared Task on Automatic Term Extraction Using the
Annotated Corpora for Term Extraction Research (ACTER) Dataset](https://www.aclweb.org/anthology/2020.computerm-1.12.pdf).

<!-- - ### __ATE difficutlties__
    - Time- and effort-consuming data collection, manual annotation
    - Extraction methodology
        - Difficult to quantify how specialises/ domain-specific a lexical unit needs to be before it is considered a term i.e  term length, term POS-pattern, minimum term frequency...
    - Evaluation methods
        - Limited to a single resource, or the calculation of precision. -->

- ### __Dataset__: Annotated Corpora for Term Extraction Research (ACTER) - v1.2
    - Descriptions:
        - Domains: Corruption, dressage, wind energy (train), heart failure (test).
        - Languages: : English, French and Dutch.
        - ~50k tokens/language/domain manually annotated 
            - Unstructured lists of all unique annotated terms.
    - Labels: term or not (binary task) 
        - Named Entities (optional)
        - True terms
        <table border="0">
            <tr>
                <td><b style="font-size:10px"></b></td>
                <td><b style="font-size:15px">Specific Terms</b></td>
                <td><b style="font-size:15px">Common Terms</b></td>
                <td><b style="font-size:15px">Out-Of-Domain Terms</b></td>
            </tr>
            <tr>
                <td>
                    <ul style="list-style-type:none;">
                        <li>Domain-specific</li>
                        <li>Lexical-specific</li>
                    </ul>
                </td>
                <td>
                    <ul style="list-style-type:none;">
                        <li>x</li>
                        <li>x</li>
                    </ul>
                </td>
                <td>
                    <ul style="list-style-type:none;">
                        <li>x</li>
                        <li>o</li>
                    </ul>
                </td>
                <td>
                    <ul style="list-style-type:none;">
                        <li>o</li>
                        <li>x</li>
                    </ul>
                </td>
            </tr>
        </table>

        
    - 2 datasets: with and without Named Entities

- ### __Evaluation Metrics__: 
    - Precision: how many of the extracted terms are correct.
    - Recall: how many of the terms in the text have correctly been extracted.
    - F1-score: harmonic mean (gold standard with only terms and with both terms and Named Entities).

- ### __Methodology__: 
    - __NYU__: English only - Termolator.
        - Select candidate terms based on chunking and abbreviations.
        - Calculate distribution metrics, well-formedness.
        - Calculate relevance score.

    - __RACAI__: English only
        - Combine several statistical approachs and vote to generate results
            - TextRank, TFIDF, clustering, termhood features.

    - __e-Terminology__: All 3 languages
        - TSR (Token Slot Recognition) technique in TBXTools
            - Dutch: statistical version
            - Enlish, French: linguistic version
        - Filter out stopwords and f(terms) <= 2
        - Terminological reference: IATE database for 12-Law.

    - __MLPLab_UQAM__:  All 3 languages
        - Bidirectional LSTM with GloVe embeddings

    - __TALN-LS2N__: only English, French (described in next paper)
        
- ### __Notes__:
    - TALN-LS2N’s system
outperforms all others in the English and French tracks.
    - NLPLab UQAM’s system outperforms e-Terminology for the Dutch track.
    - Unpredictability of DL models (BERT)
        - Large gap between precision and recall for English model, much smaller for French model.
    - [ACTER v1.3](https://clarin.eurac.edu/repository/xmlui/handle/20.500.12124/24#:~:text=The%20ACTER%20(Annotated%20Corpora%20for,failure%2C%20and%20wind%20energy).)
        - Data description in [README.html](./Data/README.html).


2. [TALN-LS2N System for Automatic Term Extraction](https://www.aclweb.org/anthology/2020.computerm-1.13.pdf).

- ### __Dataset__: Annotated Corpora for Term Extraction Research (ACTER)
    - Training phase: Corruption, dressage, wind energy.
    - Test phase: Heart failure.
    - Languages: : English, French and Dutch.

- ### __Proposed systems__:
<table border="0">
    <tr>
        <td><b style="font-size:20px">Feature-based approaches</b></td>
        <td><b style="font-size:20px">Context-based approaches</b></td>
    </tr>
    <tr>
        <td>
            <ol>
                <li>Feature Extraction</li>
                <ul>
                    <li>Linguistic filtering</li>
                    <ul>
                        <li>spaCy’s rule-matching engine </li>
                    </ul>
                    <li>Candidate describing</li>
                    <ul>
                        <li>Linguistic, stylistic, statistic, and distributional descriptors </li>
                        <li>Termhood: degree to which a linguistic unit is related to domain-specific context </li>
                        <li>Measures: Specificity, Term’s relation to Context, Cvalue, Termhood </li>
                    </ul>
                    <li>Selection phase</li>
                </ul>
                <li>Classification - Boosting</li>
                <ul>
                    <li>sklearn standard scaler</li>
                    <li>eXtreme Gradient Boosting (XGBoost)</li>
                </ul>
            </ol>
        </td>
        <td>
            <ol>
                <li>Formats:
                    <ul>
                        <li>Input: The sentence contains the term </li>
                        <li>Output: The term</li>
                    </ul>
                </li>
                <li>Models</li>
                <ul>
                    <li>English: RoBERTa</li>
                    <ul>
                        <li>Modify key hyperparams in original BERT </li>
                        <li>Eliminate its next sentence pretraining objective </li>
                        <li>Train the model with much larger mini-batches and more substantial learning rates </li>
                    </ul>
                    <li>French: CamemBERT</li>
                    <li>Use pre-trained models and fine-tuned during the classification</li>
                </ul>
            </ol>
        </td>
    </tr>
</table>

- ### __Notes__:
    - BERT outperforms classical methods
    - New, simple and strong baseline for terminology extraction
---

## Contributors:
- 🐮 [@honghanhh](https://github.com/honghanhh)