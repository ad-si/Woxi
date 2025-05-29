#!/usr/bin/env wolframscript

words = RandomSample[WordData /@ WordList[], 10];

results = words[[1 ;; 5]] //
  Map [First, #]& //
  Map [GetExampleSentence, #]&;

Print["\n-------\n"]；

results // Map [Print, #]&;
