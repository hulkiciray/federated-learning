## Masked Language Modeling (MLM)
We will probably implement FL with BERT, encoder model trained with MLM strategy. Hence, this section of the
repo is to get familiar with preparing data and building training part for masked language modeling approach.
Below you may see the language list that are used during pretraining mBERT.

| Column 1             | Column 2              | Column 3              | Column 4                   |
|----------------------|-----------------------|-----------------------|----------------------------|
| Afrikaans            | Albanian              | Arabic                | Aragonese                  |
| Armenian             | Asturian              | Azerbaijani           | Bashkir                    |
| Basque               | Bavarian              | Belarusian            | Bengali                    |
| Bishnupriya Manipuri | Bosnian               | Breton                | Bulgarian                  |
| Burmese              | Catalan               | Cebuano               | Chechen                    |
| Chinese (Simplified) | Chinese (Traditional) | Chuvash               | Croatian                   |
| Czech                | Danish                | Dutch                 | English                    |
| Estonian             | Finnish               | French                | Galician                   |
| Georgian             | German                | Greek                 | Gujarati                   |
| Haitian              | Hebrew                | Hindi                 | Hungarian                  |
| Icelandic            | Ido                   | Indonesian            | Irish                      |
| Italian              | Japanese              | Javanese              | Kannada                    |
| Kazakh               | Kirghiz               | Korean                | Latin                      |
| Latvian              | Lithuanian            | Lombard               | Low Saxon                  |
| Luxembourgish        | Macedonian            | Malagasy              | Malay                      |
| Malayalam            | Marathi               | Minangkabau           | Nepali                     |
| Newar                | Norwegian (Bokmål)    | Norwegian (Nynorsk)   | Occitan                    |
| Persian (Farsi)      | Piedmontese           | Polish                | Portuguese                 |
| Punjabi              | Romanian              | Russian               | Scots                      |
| Serbian              | Serbo-Croatian        | Sicilian              | Slovak                     |
| Slovenian            | South Azerbaijani     | Spanish               | Sundanese                  |
| Swahili              | Swedish               | Tagalog               | Tajik                      |
| Tamil                | Tatar                 | Telugu                | Turkish                    |
| Ukrainian            | Urdu                  | Uzbek                 | Vietnamese                 |
| Volapük              | Waray-Waray           | Welsh                 | West Frisian               |
| Western Punjabi      | Yoruba                | Thai (on new version) | Mongolian (on new version) |
**Note:** The Multilingual Cased (New) release contains additionally **Thai** and **Mongolian**.

---
#### Reference
* [mBERT corpora language list](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages)
* [Model types and some other information](https://huggingface.co/docs/transformers/glossary)
* [Data preperation](https://www.youtube.com/watch?v=q9NS5WpfkrU)
* [Model training](https://www.youtube.com/watch?v=R6hcxMMOrPE)
