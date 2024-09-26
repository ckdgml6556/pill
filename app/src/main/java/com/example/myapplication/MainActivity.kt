package com.example.myapplication

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import kotlin.system.measureTimeMillis
import com.example.myapplication.YOLOv8ObjectDetector as YOlOv8
import com.example.myapplication.data.DetectionResult
import com.example.myapplication.data.BoundingBox
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {

    lateinit var stage1_model: YOlOv8
    lateinit var stage2_model: YOlOv8

    private lateinit var resultView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d("Yolov8", "yolov8}")
        setContentView(R.layout.activity_main)
        stage1_model = YOlOv8(resources.assets, "od_stage1.tflite", 2)
        stage2_model = YOlOv8(resources.assets, "od_stage2.tflite", 1)

        resultView = findViewById(R.id.detectionResultView)

        detect()

    }

    fun detect(){
        val bitmap = BitmapFactory.decodeResource(resources, R.drawable.test)
//        val measuredTime = measureTimeMillis {
//            val result = stage1_model.nms(stage1_model.detectObjects(bitmap, YOlOv8.InputDataType.FLOAT32),0.4f,0.3f)
//        }
        val detectionResults = stage2_model.nms(stage2_model.detectObjects(bitmap, YOlOv8.InputDataType.FLOAT32),0.3f,0.3f)
        val resultBitmap = drawDetectionResultsOnBitmap(this, bitmap, detectionResults)
        resultView.setImageBitmap(resultBitmap)
        //detectionResultView.setDetectionResults(result, bitmap)

        //Log.d("Yolov8", "time = ${measuredTime}")

    }

    // 비트맵과 BoundingBox 모두 핸드폰 화면 가로 크기에 맞게 리사이징 후 원 그리기
    fun drawDetectionResultsOnBitmap(
        context: Context,
        bitmap: Bitmap,
        detectionResults: List<DetectionResult>
    ): Bitmap {
        // 1. 핸드폰 화면 가로 크기에 맞춰 비트맵 리사이징
        val resizedBitmap = resizeBitmapToScreenWidth(context, bitmap)

        // 2. 원본 비트맵 크기 정보
        val originalWidth = YOlOv8.inputWidth
        val originalHeight = YOlOv8.inputHeight
        val resizedWidth = resizedBitmap.width
        val resizeHeight = resizedBitmap.height

        // 3. 비트맵에 그리기 위한 Canvas 생성
        val mutableBitmap = resizedBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        // 4. Paint 객체 생성 (원의 색상, 스타일 등 정의)
        val paint = Paint().apply {
            color = Color.GREEN
            style = Paint.Style.FILL
            isAntiAlias = true  // 원의 가장자리를 부드럽게
        }

        // 5. 각 BoundingBox의 중앙에 원 그리기 (화면 가로 비율에 맞춰 조정)
        detectionResults.forEach { result ->
            // BoundingBox 리사이징
            val resizedBox = resizeBoundingBoxToScreenWidth(result.boundingBox, originalWidth, resizedWidth, originalHeight, resizeHeight)

            // 중앙값 계산
            val centerX = (resizedBox.left + resizedBox.right) / 2
            val centerY = (resizedBox.top + resizedBox.bottom) / 2

            // 캔버스에 원 그리기 (중앙에 반지름 10f인 원)
            canvas.drawCircle(centerX, centerY, 5f, paint)
        }

        return mutableBitmap
    }

    // 핸드폰의 화면 가로 크기에 맞춰 비트맵을 리사이징
    fun resizeBitmapToScreenWidth(context: Context, bitmap: Bitmap): Bitmap {
        // 화면 크기 가져오기
        val displayMetrics = context.resources.displayMetrics
        val screenWidth = displayMetrics.widthPixels

        // 원래 비트맵 크기
        val originalWidth = bitmap.width
        val originalHeight = bitmap.height

        // 가로 비율에 맞춰 세로 크기를 계산
        val aspectRatio = originalHeight.toFloat() / originalWidth.toFloat()
        val resizedHeight = (screenWidth * aspectRatio).toInt()

        // 비트맵 리사이징
        return Bitmap.createScaledBitmap(bitmap, screenWidth, resizedHeight, true)
    }

    fun resizeBoundingBoxToScreenWidth(
        box: BoundingBox,
        originalWidth: Int,
        resizedWidth: Int,
        originalHeight: Int,
        resizedHeight: Int
    ): BoundingBox {
        val widthScale = resizedWidth.toFloat() / originalWidth
        val heightScale = resizedHeight.toFloat() / originalHeight// 가로 비율을 기준으로 세로도 같은 비율로 조정

        return BoundingBox(
            left = box.left * widthScale,
            top = box.top * heightScale,
            right = box.right * widthScale,
            bottom = box.bottom * heightScale
        )
    }




}