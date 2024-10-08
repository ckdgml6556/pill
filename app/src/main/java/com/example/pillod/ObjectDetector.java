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

    public static final int CLASS_COUNT = 1;

    public static final Map<String, String> CLASSES = new HashMap<String, String>() {
        {
            put("0", "pill");
            put("1", "plate");
        }
    };

    private Interpreter interpreter;
    private int numBoxes = 8400;
    private int numClasses = 2;
    private int numCoordsPerBox = 4 + numClasses;
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

    public List<DetectionResult> detectObjects(Bitmap bitmap, InputDataType inputDataType) {
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
            float confidence = 0.0f;
            int classId = 0;
            for (int i = 0; i < CLASS_COUNT; i++) {
                if (confidence < outputArray[0][i][boxIndex]) {
                    confidence = outputArray[0][4 + i][boxIndex];
                    classId = i;
                }
            }

            if (confidence > CONFIDENCE_THRESHOLD) {
                float centerX = outputArray[0][CLASS_COUNT - 1][boxIndex];
                float centerY = outputArray[0][CLASS_COUNT][boxIndex];
                float width = outputArray[0][CLASS_COUNT + 1][boxIndex];
                float height = outputArray[0][CLASS_COUNT + 2][boxIndex];
                BoundingBox bbox = convertYOLOToBoundingBox(centerX, centerY, width, height, bitmap.getWidth(), bitmap.getHeight());
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
        List<DetectionResult> selectedDetections = new ArrayList<>();

        List<DetectionResult> filteredDetections = new ArrayList<>();
        for (DetectionResult detection : detections) {
            if (detection.getConfidence() >= confidenceThreshold) {
                filteredDetections.add(detection);
            }
        }

        filteredDetections.sort((d1, d2) -> Float.compare(d2.getConfidence(), d1.getConfidence()));

        boolean[] isSuppressed = new boolean[filteredDetections.size()];

        for (int i = 0; i < filteredDetections.size(); i++) {
            if (!isSuppressed[i]) {
                BoundingBox boxA = filteredDetections.get(i).getBoundingBox();
                selectedDetections.add(filteredDetections.get(i));

                for (int j = i + 1; j < filteredDetections.size(); j++) {
                    if (!isSuppressed[j]) {
                        BoundingBox boxB = filteredDetections.get(j).getBoundingBox();
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

    public List<DetectionResult> getDetectionResult(Bitmap bitmap,  float iouThreshold, float confidenceThreshold){
        return nms(detectObjects(bitmap, InputDataType.FLOAT32),iouThreshold, confidenceThreshold);
    }

}