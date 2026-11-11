# Blindness audit

Every tool call in the Fable session whose command or file path mentions a blinded or forbidden location, generated automatically from the session log.

READ predictions were written to disk at **2026-11-10 19:26:55 UTC**.

| Time (UTC) | Before READ commit? | Mentions | Kind | Action |
|---|---|---|---|---|
| 19:26:16 | yes | blind READ key | command failed to parse; nothing ran | Save Stage 1 predictions and run the blind 9-model evaluator |
| 19:27:03 | no | blind READ key | command | Run the blind 9-model READ evaluator and log its output |
| 19:27:23 | no | blind READ key | command | Remove the duplicated stray lines in the parser, verify it compiles, and run the READ evaluator |
| 19:27:34 | no | blind READ key | command | Run the READ evaluator with UTF-8 console output |
| 19:31:53 | no | original EDIT checkpoint | text of a document Fable wrote (no file access) | Write raw_response.txt |
| 19:31:53 | no | EDIT corruption index files | text of a document Fable wrote (no file access) | Write raw_response.txt |
| 10:07:51 | no | original EDIT checkpoint | text of a document Fable wrote (no file access) | Write final_wsp_report.md |
| 10:07:51 | no | EDIT corruption index files | text of a document Fable wrote (no file access) | Write final_wsp_report.md |

How to read this table:

- The READ key is only ever passed to the evaluator. The one occurrence before the commit time is a command that failed at the shell's parsing stage, so neither the save nor the evaluator ran; the predictions were then written and the evaluator run afterwards.
- Rows of kind 'text of a document' are the isolation statements Fable wrote in its EDIT response and final report, which name the forbidden paths in order to say they were not used.
- The checkpoint directory appears only as the name of a folder in a top-level listing; its file names were never listed by Fable.
- The EDIT index files appear only as names in a directory listing; they were never opened. The original EDIT checkpoint was never opened.
- Disclosed in the EDIT response: the public file `variant_a_gaussian_noise.json` lists the noised filters. The repair finds them from the tensor itself.
