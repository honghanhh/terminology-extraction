# Terminology Extraction - Paper Summary


1.  [Shared Task on Automatic Term Extraction Using the
Annotated Corpora for Term Extraction Research (ACTER) Dataset](https://www.aclweb.org/anthology/2020.computerm-1.12.pdf).

2. [TALN-LS2N System for Automatic Term Extraction](https://www.aclweb.org/anthology/2020.computerm-1.13.pdf).
<!-- 
- Automatic terminology extraction:
    - Rule-based approaches
        - Conducted by a terminologist
        - Hand-operated exploration, indexation, 
        - Costly maintenance of domain-specific corpora and terminologies 
    - __Feature-based approaches__
    - __Context-based approaches__
    - Hybrid approaches. -->

- __Dataset__:
    - Training phase: Corruption, dressage, wind energy.
    - Test phase: Heart failure.
    - Languages: : English, French and Dutch.

- __Proposed systems__:
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

## Contributors:
- 🐮 [@honghanhh](https://github.com/honghanhh)