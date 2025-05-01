#import "@preview/charged-ieee:0.1.3": ieee

#show: ieee.with(
    title: [BERTHA: Binary Estimator for Robust Temporal Hesitation Analysis],
    abstract: [
        This is an abstract. #lorem(50)
    ],
    authors: (
        (
            name: "Andrew Ash",
            department: [Analog VLSI Laboratory],
            organization: [Oklahoma State University],
            email: "andrew.ash@okstate.edu"
        ),
        (
            name: "Laurenz Mädje",
            department: [VLSI Computer Architecture Research Laboratory],
            organization: [Oklahoma State University],
            email: "marcus@infinitymdm.dev"
        ),
    ),
    index-terms: (),
    bibliography: bibliography("refs.bib"),
    figure-supplement: [Fig.],
)

= Introduction
Motivation & initial concept go here

== Paper Overview

= Design

== Requirements

== Dataset Selection

== Transfer Model Selection

== Performance Metrics

= Model Tuning & Training Results

== Initial Model

== Using Custom BCE Loss

== Using Weighted BCE Loss

== Adding Data Augmentation

== Adding a Dense Layer

== Adjusting Decision Thresholds

= Real-time Filler Detection

= Conclusion
Improvements and recommendations go here
