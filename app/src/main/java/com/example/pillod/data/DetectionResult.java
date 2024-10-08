package com.example.pillod.data;

public class DetectionResult {
    private int classId;
    private float confidence;
    private BoundingBox boundingBox;

    public DetectionResult(int classId, float confidence, BoundingBox boundingBox) {
        this.classId = classId;
        this.confidence = confidence;
        this.boundingBox = boundingBox;
    }

    public int getClassId() {
        return classId;
    }

    public void setClassId(int classId) {
        this.classId = classId;
    }

    public float getConfidence() {
        return confidence;
    }

    public void setConfidence(float confidence) {
        this.confidence = confidence;
    }

    public BoundingBox getBoundingBox() {
        return boundingBox;
    }

    public void setBoundingBox(BoundingBox boundingBox) {
        this.boundingBox = boundingBox;
    }

    @Override
    public String toString() {
        return "DetectionResult{" +
                "classId=" + classId +
                ", confidence=" + confidence +
                ", boundingBox=" + boundingBox +
                '}';
    }
}