#import "@preview/kunskap:0.1.0": *

#show: kunskap.with(
    title: [Final Project Proposal],
    author: "Andrew Ash and Marcus Mellor",
    header: "ECEN5060 Final Project",
    date: datetime.today().display("[month repr:long] [day padding:zero], [year repr:full]"),
)

= Proposed Concept: Automated "Ah" Counter

We propose a deep neural network designed to replicate the role of an "ah" counter, as in public
speaking organizations such as Toastmasters. The network should take real-time audio as input and
indicate whether the most recent second of audio contains filler words such as "ah" or "um".

= Detailed Design

== Functional Requirements

== Training Dataset

The TED-LIUM dataset is a corpus of English-language TED talk audio with transcriptions. Crucially
for our application, these transcripts include speech disfluencies such as filler words. The
dataset was originally developed to help train speech recognition tasks.

The dataset consists of 452 hours of audio sampled at 16kHz with verbatim transcriptions in the
text-based STM format. Filler words are mapped to the text `{FILLX}` where `X` is replaced with a
numeral. Transcripts are aligned with audio using the Kaldi toolkit.

This transcription scheme should allow us to train on aligned audio and identify whether the last
second contained a filler word. The dataset provides a suggested split into training, validation,
and test sets. We plan to use the same split.

== Network Architecture and Performance Metrics

== Schedule

