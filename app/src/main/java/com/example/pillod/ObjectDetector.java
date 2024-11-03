package com.example.pillod;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

import com.example.pillod.data.BoundingBox;
import com.example.pillod.data.DetectionResult;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.lang.Math.max;
import static java.lang.Math.min;

public class ObjectDetector {

    public enum InputDataType {
        FLOAT32,
        FLOAT16
    }

    public static final float CONFIDENCE_THRESHOLD = 0.3f;
    public static final float IMAGE_MEAN = 127.5f;
    public static final float IMAGE_STD = 127.5f;

    public static final int inputWidth = 640;
    public static final int inputHeight = 640;

    public static int CLASS_PILL = 0;
    public static int CLASS_PLATE = 1;

    private Interpreter interpreter;
    private int numBoxes = 8400;
    private int numClasses = 2;
    private int numCoordsPerBox = 4 + numClasses;
    private int limitDetectionNum = 1000;
    private int bufferSize;
    private ByteBuffer byteBuffer;

    public ObjectDetector(AssetManager assetManager, String modelPath, int classNum) {
        Interpreter.Options options = new Interpreter.Options();
        interpreter = new Interpreter(loadModelFile(assetManager, modelPath), options);

        numClasses = classNum;
        numCoordsPerBox = 4 + classNum;

        bufferSize = 4 * inputWidth * inputHeight * 3; // Float32: 4 bytes per float
        byteBuffer = ByteBuffer.allocateDirect(bufferSize);
        byteBuffer.order(ByteOrder.nativeOrder());
    }

    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) {
        try {
            FileInputStream inputStream = new FileInputStream(assetManager.openFd(modelPath).getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = assetManager.openFd(modelPath).getStartOffset();
            long declaredLength = assetManager.openFd(modelPath).getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } catch (Exception e) {
            throw new RuntimeException("Error loading model file", e);
        }
    }

    public List<DetectionResult> detectObjects(Bitmap bitmap, InputDataType inputDataType, BoundingBox inputBox) {
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true);
        ByteBuffer inputArray;

        switch (inputDataType) {
            case FLOAT32:
                inputArray = convertBitmapToByteBufferFloat32(scaledBitmap);
                break;
            case FLOAT16:
                inputArray = convertBitmapToByteBufferFloat16(scaledBitmap);
                break;
            default:
                throw new IllegalArgumentException("Invalid InputDataType");
        }

        float[][][] outputArray = new float[1][numCoordsPerBox][numBoxes];
        interpreter.run(inputArray, outputArray);

        List<DetectionResult> detectionResults = new ArrayList<>();
        for (int boxIndex = 0; boxIndex < numBoxes; boxIndex++) {
            if (detectionResults.size() >= limitDetectionNum) { // 검출 개수 제한 추가
                break;
            }

            float confidence = 0.0f;
            int classId = 0;
            for (int i = 0; i < numClasses; i++) {
                float classConfidence = outputArray[0][4 + i][boxIndex];
                if (confidence < classConfidence) {
                    confidence = classConfidence;
                    classId = i;
                }
            }

            // PCH : Bug수정
            if (confidence > CONFIDENCE_THRESHOLD) {
                float centerX = outputArray[0][0][boxIndex];
                float centerY = outputArray[0][1][boxIndex];
                float width = outputArray[0][2][boxIndex];
                float height = outputArray[0][3][boxIndex];
                BoundingBox bbox = convertYOLOToBoundingBox(centerX, centerY, width, height, bitmap.getWidth(), bitmap.getHeight());
                // PCH : 이미지를 Crop하여 넣었을때 좌표 보정 처리하기 위한 방법
                bbox.setLeft(bbox.getLeft() > 0 ? bbox.getLeft() + inputBox.getLeft() : 0);
                bbox.setRight(bbox.getRight() < bitmap.getWidth() ? bbox.getRight() + inputBox.getLeft() : bitmap.getWidth());
                bbox.setTop(bbox.getTop() > 0 ? bbox.getTop() + inputBox.getTop() : 0);
                bbox.setBottom(bbox.getBottom() < bitmap.getHeight()? bbox.getBottom() + inputBox.getTop() : bitmap.getHeight());

                detectionResults.add(new DetectionResult(classId, confidence, bbox));
            }
        }
        return detectionResults;
    }

    private ByteBuffer convertBitmapToByteBufferFloat32(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bufferSize);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[inputWidth * inputHeight];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputWidth; i++) {
            for (int j = 0; j < inputHeight; j++) {
                int value = intValues[pixel++];
                byteBuffer.putFloat(((value >> 16 & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat(((value >> 8 & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat(((value & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }

        byteBuffer.rewind();
        return byteBuffer;
    }

    private ByteBuffer convertBitmapToByteBufferFloat16(Bitmap bitmap) {
        ShortBuffer shortBuffer = ShortBuffer.allocate(inputWidth * inputHeight * 3);

        int[] intValues = new int[inputWidth * inputHeight];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputWidth; i++) {
            for (int j = 0; j < inputHeight; j++) {
                int value = intValues[pixel++];
                shortBuffer.put((short) (((value >> 16 & 0xFF) - IMAGE_MEAN)));
                shortBuffer.put((short) (((value >> 8 & 0xFF) - IMAGE_MEAN)));
                shortBuffer.put((short) (((value & 0xFF) - IMAGE_MEAN)));
            }
        }

        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(shortBuffer.capacity() * 2);
        byteBuffer.order(ByteOrder.nativeOrder());
        shortBuffer.rewind();
        byteBuffer.asShortBuffer().put(shortBuffer);

        byteBuffer.rewind();
        return byteBuffer;
    }

    private BoundingBox convertYOLOToBoundingBox(float centerX, float centerY, float width, float height, int imageWidth, int imageHeight) {
        float left = (centerX - width / 2.0f) * imageWidth;
        float top = (centerY - height / 2.0f) * imageHeight;
        float right = (centerX + width / 2.0f) * imageWidth;
        float bottom = (centerY + height / 2.0f) * imageHeight;
        return new BoundingBox(left, top, right, bottom);
    }

    public List<DetectionResult> nms(List<DetectionResult> detections, float iouThreshold, float confidenceThreshold) {
        // 클래스별로 DetectionResult를 분류
        Map<Integer, List<DetectionResult>> classwiseDetections = new HashMap<>();
        int count = 0;
        for (DetectionResult detection : detections) {
            if (detection.getConfidence() >= confidenceThreshold) {
                int classId = detection.getClassId();  // 클래스 ID를 가져옴
                classwiseDetections.putIfAbsent(classId, new ArrayList<>());
                classwiseDetections.get(classId).add(detection);
            }
        }

        // 각 클래스별로 NMS 수행
        List<DetectionResult> selectedDetections = new ArrayList<>();
        for (List<DetectionResult> classDetections : classwiseDetections.values()) {
            selectedDetections.addAll(performNMSForSingleClass(classDetections, iouThreshold));
        }

        return selectedDetections;
    }

    private List<DetectionResult> performNMSForSingleClass(List<DetectionResult> detections, float iouThreshold) {
        List<DetectionResult> selectedDetections = new ArrayList<>();

        // 신뢰도에 따라 내림차순으로 정렬
        detections.sort((d1, d2) -> Float.compare(d2.getConfidence(), d1.getConfidence()));

        boolean[] isSuppressed = new boolean[detections.size()];

        // NMS 수행
        for (int i = 0; i < detections.size(); i++) {
            if (!isSuppressed[i]) {
                BoundingBox boxA = detections.get(i).getBoundingBox();
                selectedDetections.add(detections.get(i));

                for (int j = i + 1; j < detections.size(); j++) {
                    if (!isSuppressed[j]) {
                        BoundingBox boxB = detections.get(j).getBoundingBox();
                        float iou = calculateIOU(boxA, boxB);

                        if (iou >= iouThreshold) {
                            isSuppressed[j] = true;
                        }
                    }
                }
            }
        }

        return selectedDetections;
    }

    public float calculateIOU(BoundingBox boxA, BoundingBox boxB) {
        float intersectionLeft = max(boxA.getLeft(), boxB.getLeft());
        float intersectionTop = max(boxA.getTop(), boxB.getTop());
        float intersectionRight = min(boxA.getRight(), boxB.getRight());
        float intersectionBottom = min(boxA.getBottom(), boxB.getBottom());

        float intersectionArea = max(0.0f, intersectionRight - intersectionLeft) * max(0.0f, intersectionBottom - intersectionTop);

        float areaA = (boxA.getRight() - boxA.getLeft()) * (boxA.getBottom() - boxA.getTop());
        float areaB = (boxB.getRight() - boxB.getLeft()) * (boxB.getBottom() - boxB.getTop());

        return intersectionArea / (areaA + areaB - intersectionArea);
    }

    public List<DetectionResult>  getResults(Bitmap bitmap, float iouThreshold, float confidenceThreshold, BoundingBox bbox){
        return nms(detectObjects(bitmap, InputDataType.FLOAT32, bbox),iouThreshold, confidenceThreshold);
    }

    public int getClassNum(){
        return numClasses;
    }
}