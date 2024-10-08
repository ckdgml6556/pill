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

    private static final float IOU_THRESHOLD = 0.5f;
    private static final float CONF_THRESHOLD = 0.4f;
    private ObjectDetector stage1Detector;
    private ObjectDetector stage2Detector;
    private ImageView resultView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        stage1Detector = new ObjectDetector(getAssets(), "od_stage1.tflite", 2);
        stage2Detector = new ObjectDetector(getAssets(), "od_stage2.tflite", 1);

        resultView = findViewById(R.id.detectionResultView);

        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.test);

        detect(bitmap);
    }

    private void detect(Bitmap bitmap) {
        BoundingBox cropBox = processStage1(stage1Detector, bitmap);
        List<DetectionResult> detectionResults = processStage2(stage2Detector, bitmap, cropBox);
        Bitmap resultBitmap = drawDetectionResultsOnBitmap(this, bitmap, detectionResults, cropBox);
        resultView.setImageBitmap(resultBitmap);
    }

    private BoundingBox processStage1(ObjectDetector detector, Bitmap bitmap) {
        // 플레이트 검출이 불가능한 모델을 넣은 경우
        if (detector.getClassNum() == 1) {
            return new BoundingBox(0.0f, 0.0f, bitmap.getWidth(), bitmap.getHeight());
        }
        List<DetectionResult> detectionResults = detector.getResults(bitmap, IOU_THRESHOLD, CONF_THRESHOLD, new BoundingBox(0.0f, 0.0f, bitmap.getWidth(), bitmap.getHeight()));
        /*
            플레이트 검출 조건 : 플레이트 안에 약이 있는 플레이트만 검출하여 바운딩 박스를 합쳐서 최종 Crop Size를 정함. 없으면 전체 이미지.
         */
        BoundingBox cropBox = null;
        for (DetectionResult dr : detectionResults) {
            if (dr.getClassId() == ObjectDetector.CLASS_PLATE) {
                BoundingBox plateBox = dr.getBoundingBox();
                for (DetectionResult comp_dr : detectionResults) {
                    if (comp_dr.getClassId() == ObjectDetector.CLASS_PILL) {
                        BoundingBox pillBox = comp_dr.getBoundingBox();
                        float pillCenterX = pillBox.getRight() + pillBox.getLeft() / 2;
                        float pillCenterY = pillBox.getTop() + pillBox.getBottom() / 2;
                        if (plateBox.getLeft() < pillCenterX & plateBox.getRight() > pillCenterX &
                                plateBox.getTop() < pillCenterY & plateBox.getBottom() > pillCenterY){
                            if (cropBox == null) {
                                cropBox = plateBox;
                            } else {
                                if (cropBox.getLeft() > plateBox.getLeft())
                                    cropBox.setLeft(plateBox.getLeft());
                                if (cropBox.getTop() > plateBox.getTop())
                                    cropBox.setTop(plateBox.getTop());
                                if (cropBox.getRight() < plateBox.getRight())
                                    cropBox.setRight(plateBox.getRight());
                                if (cropBox.getBottom() < plateBox.getBottom())
                                    cropBox.setBottom(plateBox.getBottom());
                            }
                            break;
                        }
                    }
                }
            }
        }
        return cropBox == null ? new BoundingBox(0.0f, 0.0f, bitmap.getWidth(), bitmap.getHeight()) : cropBox;
    }

    private List<DetectionResult> processStage2(ObjectDetector detector, Bitmap bitmap, BoundingBox cropBox) {
        int cropStartX =(int)cropBox.getLeft();
        int cropStartY = (int)cropBox.getTop();
        int cropWidth = (int)(cropBox.getRight()-cropBox.getLeft());
        int cropHeight = (int)(cropBox.getBottom()-cropBox.getTop());
        if(cropStartX + cropWidth >= bitmap.getWidth())
            cropWidth = bitmap.getWidth() - cropStartX;
        if(cropStartY + cropHeight >= bitmap.getWidth())
            cropHeight = bitmap.getHeight() - cropStartY;
        Bitmap cropBitmap = Bitmap.createBitmap(bitmap, cropStartX, cropStartY, cropWidth, cropHeight);
        return detector.getResults(cropBitmap, IOU_THRESHOLD, CONF_THRESHOLD, cropBox);
    }

    // 비트맵과 BoundingBox 모두 핸드폰 화면 가로 크기에 맞게 리사이징 후 원 그리기
    private Bitmap drawDetectionResultsOnBitmap(Context context, Bitmap bitmap, List<DetectionResult> detectionResults, BoundingBox cropBox) {
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
        Paint paint_green = new Paint();
        paint_green.setColor(Color.GREEN);
        paint_green.setStyle(Paint.Style.FILL);
        paint_green.setAntiAlias(true);  // 원의 가장자리를 부드럽게

        Paint paint_red = new Paint();
        paint_red.setColor(Color.RED);
        paint_red.setStyle(Paint.Style.STROKE);
        paint_red.setAntiAlias(true);

        // 5. 각 BoundingBox의 중앙에 원 그리기 (화면 가로 비율에 맞춰 조정)
        for (DetectionResult result : detectionResults) {
            // BoundingBox 리사이징
            BoundingBox resizedBox = resizeBoundingBoxToScreenWidth(result.getBoundingBox(), originalWidth, resizedWidth, originalHeight, resizedHeight);

                // 중앙값 계산
            float centerX = (resizedBox.getLeft() + resizedBox.getRight()) / 2;
            float centerY = (resizedBox.getTop() + resizedBox.getBottom()) / 2;

            canvas.drawCircle(centerX, centerY, 5f, paint_green);
        }

        // 캔버스에 원 그리기 (중앙에 반지름 10f인 원)
        BoundingBox resizeCropBox = resizeBoundingBoxToScreenWidth(cropBox, originalWidth, resizedWidth, originalHeight, resizedHeight);
        canvas.drawRect(resizeCropBox.getLeft(), resizeCropBox.getTop(), resizeCropBox.getRight(), resizeCropBox.getBottom(), paint_red);

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