package com.example.myapplication

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import org.tensorflow.lite.DataType
import java.nio.ShortBuffer
import kotlin.math.max
import kotlin.math.min
import com.example.myapplication.data.DetectionResult
import com.example.myapplication.data.BoundingBox


class YOLOv8ObjectDetector(assetManager: AssetManager, modelPath: String, classNum: Int) {

    enum class InputDataType {
        FLOAT32,
        FLOAT16
    }

    private var interpreter: Interpreter
    private val numBoxes = 8400
    private var numClasses = 2
    private var numCoordsPerBox = 4 + numClasses
    private val bufferSize: Int
    private val byteBuffer: ByteBuffer

    init {
        val options = Interpreter.Options()
        interpreter = Interpreter(loadModelFile(assetManager, modelPath), options)

        numClasses = classNum
        numCoordsPerBox = 4 + classNum

        bufferSize = 4 * inputWidth * inputHeight * 3 // Float32: 4 bytes per float
        byteBuffer = ByteBuffer.allocateDirect(bufferSize)
        byteBuffer.order(ByteOrder.nativeOrder())
    }

    private fun loadModelFile(assetManager: AssetManager, modelPath: String): ByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun detectObjects(bitmap: Bitmap, inputDataType: InputDataType): List<DetectionResult> {
        // Load the input image
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)

        // Preprocess and convert the input image to ByteBuffer
        val inputArray = when (inputDataType) {
            InputDataType.FLOAT32 -> convertBitmapToByteBufferFloat32(scaledBitmap)
            InputDataType.FLOAT16 -> convertBitmapToByteBufferFloat16(scaledBitmap)
        }

        // Run inference
        val outputArray = Array(1){Array(numCoordsPerBox) { FloatArray(numBoxes) } }// 모델의 출력 형태 변경
        interpreter.run(inputArray, outputArray)

        // Post-processing: parse the output to get detection results
        val detectionResults = mutableListOf<DetectionResult>()
        for (boxIndex in 0 until numBoxes) {
            var confidence = 0.0f
            var classId = 0
            for(i in 0 until CLASS_COUNT){
                if(confidence < outputArray[0][i][boxIndex]) {
                    confidence = outputArray[0][4+i][boxIndex]
                    classId = i
                }
            }
//            val confidence = outputArray[0][0][boxIndex] // 출력 배열의 차원 변경에 따른 수정
            if (confidence > CONFIDENCE_THRESHOLD) {
                val centerX = outputArray[0][CLASS_COUNT-1][boxIndex]
                val centerY = outputArray[0][CLASS_COUNT][boxIndex]
                val width = outputArray[0][CLASS_COUNT+1][boxIndex]
                val height = outputArray[0][CLASS_COUNT+2][boxIndex]
                // Convert YOLO format to bounding box coordinates
                val bbox = convertYOLOToBoundingBox(centerX, centerY, width, height, inputWidth, inputHeight)
                detectionResults.add(DetectionResult(classId, confidence, bbox))
            }
        }
        return detectionResults
    }

    private fun convertBitmapToByteBufferFloat32(bitmap: Bitmap): ByteBuffer {
        // Allocate ByteBuffer for float32
        val byteBuffer = ByteBuffer.allocateDirect(bufferSize)
        byteBuffer.order(ByteOrder.nativeOrder())

        // Normalize and convert bitmap pixels to ByteBuffer
        val intValues = IntArray(inputWidth * inputHeight)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until inputWidth) {
            for (j in 0 until inputHeight) {
                val value = intValues[pixel++]
                byteBuffer.putFloat(((value shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                byteBuffer.putFloat(((value shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                byteBuffer.putFloat(((value and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
            }
        }

        // Reset position to the beginning of ByteBuffer
        byteBuffer.rewind()
        return byteBuffer
    }

    private fun convertBitmapToByteBufferFloat16(bitmap: Bitmap): ByteBuffer {
        // Allocate ByteBuffer for float16
        val shortBuffer = ShortBuffer.allocate(inputWidth * inputHeight * 3)

        // Normalize and convert bitmap pixels to ShortBuffer
        val intValues = IntArray(inputWidth * inputHeight)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until inputWidth) {
            for (j in 0 until inputHeight) {
                val value = intValues[pixel++]
                shortBuffer.put(((value shr 16 and 0xFF) - IMAGE_MEAN).toInt().toShort())
                shortBuffer.put(((value shr 8 and 0xFF) - IMAGE_MEAN).toInt().toShort())
                shortBuffer.put(((value and 0xFF) - IMAGE_MEAN).toInt().toShort())
            }
        }

        // Convert ShortBuffer to ByteBuffer
        val byteBuffer = ByteBuffer.allocateDirect(shortBuffer.capacity() * 2) // 2 bytes per short
        byteBuffer.order(ByteOrder.nativeOrder())
        shortBuffer.rewind()
        byteBuffer.asShortBuffer().put(shortBuffer)

        // Reset position to the beginning of ByteBuffer
        byteBuffer.rewind()
        return byteBuffer
    }

    private fun convertYOLOToBoundingBox(centerX: Float, centerY: Float, width: Float, height: Float, imageWidth: Int, imageHeight: Int): BoundingBox {
        val xPos = (centerX - width / 2.0f) * imageWidth
        val yPos = (centerY - height / 2.0f) * imageHeight
        val w = width * imageWidth
        val h = height * imageHeight
        val left = xPos - w / 2.0f
        val top = yPos - h / 2.0f
        val right = left + w
        val bottom = top + h
        return BoundingBox(left, top, right, bottom)
    }

    fun nms(detections: List<DetectionResult>, iouThreshold: Float, confidenceThreshold: Float): List<DetectionResult> {
        // 필터링된 객체 감지 결과를 저장할 리스트
        val selectedDetections = mutableListOf<DetectionResult>()

        // 신뢰도에 따라 객체 감지 결과를 필터링
        val filteredDetections = detections.filter { it.confidence >= confidenceThreshold }

        // 신뢰도가 높은 순으로 객체 감지 결과를 정렬
        val sortedDetections = filteredDetections.sortedByDescending { it.confidence }

        // NMS 알고리즘 적용
        val isSuppressed = BooleanArray(sortedDetections.size) { false }

        for (i in sortedDetections.indices) {
            if (!isSuppressed[i]) {
                val boxA = sortedDetections[i].boundingBox
                selectedDetections.add(sortedDetections[i])

                for (j in (i + 1) until sortedDetections.size) {
                    if (!isSuppressed[j]) {
                        val boxB = sortedDetections[j].boundingBox
                        val iou = calculateIOU(boxA, boxB)

                        if (iou >= iouThreshold) {
                            isSuppressed[j] = true
                        }
                    }
                }
            }
        }

        return selectedDetections
    }

    fun calculateIOU(boxA: BoundingBox, boxB: BoundingBox): Float {
        val intersectionLeft = max(boxA.left, boxB.left)
        val intersectionTop = max(boxA.top, boxB.top)
        val intersectionRight = min(boxA.right, boxB.right)
        val intersectionBottom = min(boxA.bottom, boxB.bottom)

        val intersectionArea = max(0.0f, intersectionRight - intersectionLeft) * max(0.0f, intersectionBottom - intersectionTop)

        val areaA = (boxA.right - boxA.left) * (boxA.bottom - boxA.top)
        val areaB = (boxB.right - boxB.left) * (boxB.bottom - boxB.top)

        val iou = intersectionArea / (areaA + areaB - intersectionArea)
        return iou
    }

    companion object {
        private const val CONFIDENCE_THRESHOLD = 0.3f
        private const val IMAGE_MEAN = 127.5f
        private const val IMAGE_STD = 127.5f

        val inputWidth = 640
        val inputHeight = 640

        private const val CLASS_COUNT = 1
        val CLASSES:Map<String, String> = mapOf(
            "0" to "pill",
            "1" to "plate"
        )
    }
}
