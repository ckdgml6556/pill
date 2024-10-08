package com.example.pillod;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import com.example.pillod.data.BoundingBox;
import com.example.pillod.data.DetectionResult;

import java.util.List;

public class MainActivity extends AppCompatActivity {

    private ObjectDetector stage1_detector;
    private ObjectDetector stage2_detector;

    private ImageView resultView;

    private static final float IOU_THRESHOLD = 0.4f;
    private static final float CONF_THRESHOLD = 0.3f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d("Yolov8", "yolov8}");
        setContentView(R.layout.activity_main);
        stage1_detector = new ObjectDetector(getAssets(), "od_stage1.tflite", 2);
        stage2_detector = new ObjectDetector(getAssets(), "od_stage2.tflite", 1);

        resultView = findViewById(R.id.detectionResultView);

        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.test2);

        detect(stage2_detector, bitmap);
    }

    private void detect(ObjectDetector detector, Bitmap bitmap) {
        // 추론 시간 체크시 사용
        // long measuredTime = measureTimeMillis(() -> {
        //     List<DetectionResult> result = stage1_model.nms(stage1_model.detectObjects(bitmap, YOLOv8ObjectDetector.InputDataType.FLOAT32), 0.4f, 0.3f);
        // });

        // Log.d("Yolov8", "time = " + measuredTime);

        List<DetectionResult> detectionResults = detector.getDetectionResult(bitmap, IOU_THRESHOLD, CONF_THRESHOLD);
        Bitmap resultBitmap = drawDetectionResultsOnBitmap(this, bitmap, detectionResults);
        resultView.setImageBitmap(resultBitmap);
    }

    // 비트맵과 BoundingBox 모두 핸드폰 화면 가로 크기에 맞게 리사이징 후 원 그리기
    private Bitmap drawDetectionResultsOnBitmap(Context context, Bitmap bitmap, List<DetectionResult> detectionResults) {
        // 1. 핸드폰 화면 가로 크기에 맞춰 비트맵 리사이징
        Bitmap resizedBitmap = resizeBitmapToScreenWidth(context, bitmap);

        // 2. 원본 비트맵 크기 정보
        int originalWidth = bitmap.getWidth();
        int originalHeight = bitmap.getHeight();
        int resizedWidth = resizedBitmap.getWidth();
        int resizedHeight = resizedBitmap.getHeight();

        // 3. 비트맵에 그리기 위한 Canvas 생성
        Bitmap mutableBitmap = resizedBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);

        // 4. Paint 객체 생성 (원의 색상, 스타일 등 정의)
        Paint paint = new Paint();
        paint.setColor(Color.GREEN);
        paint.setStyle(Paint.Style.FILL);
        paint.setAntiAlias(true);  // 원의 가장자리를 부드럽게

        // 5. 각 BoundingBox의 중앙에 원 그리기 (화면 가로 비율에 맞춰 조정)
        for (DetectionResult result : detectionResults) {
            // BoundingBox 리사이징
            BoundingBox resizedBox = resizeBoundingBoxToScreenWidth(result.getBoundingBox(), originalWidth, resizedWidth, originalHeight, resizedHeight);

            // 중앙값 계산
            float centerX = (resizedBox.getLeft() + resizedBox.getRight()) / 2;
            float centerY = (resizedBox.getTop() + resizedBox.getBottom()) / 2;

            // 캔버스에 원 그리기 (중앙에 반지름 10f인 원)
            canvas.drawCircle(centerX, centerY, 5f, paint);
        }

        return mutableBitmap;
    }

    // 핸드폰의 화면 가로 크기에 맞춰 비트맵을 리사이징
    private Bitmap resizeBitmapToScreenWidth(Context context, Bitmap bitmap) {
        // 화면 크기 가져오기
        int screenWidth = context.getResources().getDisplayMetrics().widthPixels;

        // 원래 비트맵 크기
        int originalWidth = bitmap.getWidth();
        int originalHeight = bitmap.getHeight();

        // 가로 비율에 맞춰 세로 크기를 계산
        float aspectRatio = (float) originalHeight / originalWidth;
        int resizedHeight = (int) (screenWidth * aspectRatio);

        // 비트맵 리사이징
        return Bitmap.createScaledBitmap(bitmap, screenWidth, resizedHeight, true);
    }

    private BoundingBox resizeBoundingBoxToScreenWidth(BoundingBox box, int originalWidth, int resizedWidth, int originalHeight, int resizedHeight) {
        float widthScale = (float) resizedWidth / originalWidth;
        float heightScale = (float) resizedHeight / originalHeight; // 가로 비율을 기준으로 세로도 같은 비율로 조정

        return new BoundingBox(
                box.getLeft() * widthScale,
                box.getTop() * heightScale,
                box.getRight() * widthScale,
                box.getBottom() * heightScale
        );
    }
}