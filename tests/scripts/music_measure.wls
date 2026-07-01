(* MusicTimeSignature / MusicRest canonicalize to their association forms. *)
Print[MusicTimeSignature[3, 4]]
Print[MusicRest[1/2]]

(* A MusicMeasure resolves its events' rhythm against the meter: each event is
   annotated with its BeatDuration and Beats, and the trailing default note is
   stretched to fill the bar. *)
Print[MusicMeasure[
  {MusicNote["E", 1/2], MusicNote["D"]},
  MusicTimeSignature[4, 4]
]]

(* A compound meter (6/8) beats in dotted quarters: two beats per measure. *)
Print[MusicMeasure[
  {MusicNote["E"], MusicNote["C"]},
  MusicTimeSignature[6, 8]
]]

(* A MusicVoice packs its measures into a MeasureList association, taking its
   time signature from the first measure. *)
Print[MusicVoice[{MusicMeasure[{MusicNote["C"], MusicNote["E"]}]}]]

(* A MusicScore packs its voices into a VoiceList association. *)
Print[MusicScore[{MusicVoice[{MusicNote["C"]}]}]]
