package com.example.myapplication.data

data class DetectionResult(val classId: Int, val confidence: Float, val boundingBox: BoundingBox)