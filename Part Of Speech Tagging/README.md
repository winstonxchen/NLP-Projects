
# POS Tagging on input sentences using Python and HMM 

**Technology:Python<br>
• Library used: nltk,numpy<br>
• Stanford POS Tagger (full Stanford Tagger version 3.6.0 [124 MB]): https://nlp.stanford.edu/static/software/tagger.shtml<br>
• Implemented the hidden markov models to peform POS tagging on the input sentences and compare it to the Stanford POS Tagger.
<br>**

Steps:
1.	Print Start Probabilities (pi), Matrix A Transition Probabilities, Matrix B Observation Likelihood: Emission Probabilities.
2. Convert S1, S2 into arrays of tokens and calculate transition and emission probabilities of HMM created in 3-time steps.
3. Perform Part-of-Speech Tagging on S1, S2 and assign the probability of tagging the S1, S2 and print out the Viterbi table for S1, S2.
4. Apply Stanford POS Tagger to the sentences and output the POS tags.

