package com.example.gamecoverclassifier

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Camera
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.view.SurfaceView
import android.view.View
import android.view.WindowManager
import android.widget.ImageView
import android.widget.TextView
import android.widget.ToggleButton
import androidx.appcompat.app.AppCompatActivity
import androidx.core.graphics.createBitmap
import androidx.core.graphics.rotationMatrix
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.core.CvType.CV_32F
import org.opencv.core.CvType.CV_8UC1
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Imgproc.*
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.*

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    private var openCvCameraView : CameraBridgeViewBase? = null

    private lateinit var gray : Mat
    private lateinit var canny : Mat
    private lateinit var lines : Mat
    private lateinit var srcTransform : Mat
    private lateinit var dstTransform : Mat
    private lateinit var transformed : Mat
    private lateinit var transform : Mat
    private lateinit var previewMat : Mat
    private lateinit var template3DS : Mat
    private lateinit var templateDS : Mat
    private lateinit var tmpMat : Mat

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_IMMERSIVE or
                                                View.SYSTEM_UI_FLAG_FULLSCREEN or
                                                View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
        window.addFlags((WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON))

        OpenCVLoader.initDebug()

        gray = Mat()
        canny = Mat()
        lines = Mat()
        previewMat = Mat()
        srcTransform = Mat(4, 2, CV_32F)
        dstTransform = Mat(4, 2, CV_32F)
        transformed = Mat()
        transform = Mat()
        template3DS = Mat()
        templateDS = Mat()
        tmpMat = Mat()

        dstTransform.put(0, 0, 0.0, 0.0, 500.0, 0.0, 500.0, 500.0, 0.0, 500.0)

        openCvCameraView = findViewById<CameraBridgeViewBase>(R.id.javaCameraView)

        openCvCameraView?.setCameraPermissionGranted()
        openCvCameraView?.visibility = SurfaceView.VISIBLE
        openCvCameraView?.setCameraIndex(0)
        openCvCameraView?.setCvCameraViewListener(this)

        val bitOpts = BitmapFactory.Options()
        bitOpts.inScaled = false

        val bitmap3DS = BitmapFactory.decodeResource(resources, R.drawable.template3ds, bitOpts)
        Utils.bitmapToMat(bitmap3DS, template3DS)
        cvtColor(template3DS, template3DS, COLOR_RGBA2GRAY)

        val bitmapDS = BitmapFactory.decodeResource(resources, R.drawable.templateds, bitOpts)
        Utils.bitmapToMat(bitmapDS, templateDS)
        cvtColor(templateDS, templateDS, COLOR_RGBA2GRAY)
    }

    override fun onPause() {
        super.onPause()
        openCvCameraView?.disableView()
    }

    override fun onResume() {
        super.onResume()
        openCvCameraView?.enableView()
    }

    override fun onDestroy() {
        super.onDestroy()
        openCvCameraView?.disableView()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
    }

    override fun onCameraViewStopped() {
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        if (inputFrame != null) {
            blur(inputFrame.gray(), gray, Size(7.0, 7.0))
            Canny(gray, canny, 25.0, 75.0)

            //Remove edge on bottom of image produced from Canny
            line(canny,
                    Point(0.0, (canny.rows() - 1).toDouble()),
                    Point((canny.cols() - 1).toDouble(), (canny.rows() - 1).toDouble()),
                    Scalar(0.0), 6)

            HoughLinesP(canny, lines, 1.0, PI / 180, 50, 50.0, 49.0)

            //These represent the game box's co-ordinates
            val bX = DoubleArray(4) {canny.cols().toDouble() / 2}
            val bY = DoubleArray(4) {canny.rows().toDouble() / 2}

            //Choose points from lines to minimise distance to image corners
            for (i in 1 until lines.rows()) {
                val ps = Array<Point>(2) {j -> Point(lines[i, 0][2 * j], lines[i, 0][(2 * j) + 1])}
                for (p in ps) {
                    if ((0 - p.x).pow(2) + (0 - p.y).pow(2) <
                            (0 - bX[0]).pow(2) + (0 - bY[0]).pow(2)) {
                        bX[0] = p.x
                        bY[0] = p.y
                    }
                    if ((canny.cols() - p.x).pow(2) + (0 - p.y).pow(2) <
                            (canny.cols() - bX[1]).pow(2) + (0 - bY[1]).pow(2)) {
                        bX[1] = p.x
                        bY[1] = p.y
                    }
                    if ((canny.cols() - p.x).pow(2) + (canny.rows() - p.y).pow(2) <
                            (canny.cols() - bX[2]).pow(2) + (canny.rows() - bY[2]).pow(2)) {
                        bX[2] = p.x
                        bY[2] = p.y
                    }
                    if ((0 - p.x).pow(2) + (canny.rows() - p.y).pow(2) <
                            (0 - bX[3]).pow(2) + (canny.rows() - bY[3]).pow(2)) {
                        bX[3] = p.x
                        bY[3] = p.y
                    }
                }
            }

            inputFrame.rgba().copyTo(previewMat)

            for (i in 0 until 4) {
                val p0 = Point(bX[i], bY[i])
                val j = (i + 1) % 4
                val p1 = Point(bX[j], bY[j])
                line(previewMat, p0, p1, Scalar(0.0, 255.0, 0.0), 3, LINE_AA)
            }

            //Warp image to get perspective from top of game box
            srcTransform.put(0, 0, bX[0], bY[0], bX[1], bY[1], bX[2], bY[2], bX[3], bY[3])
            transform = getPerspectiveTransform(srcTransform, dstTransform)
            warpPerspective(gray, transformed, transform, Size(500.0, 500.0))

            //Assuming game box is upright, the N3DS logo is always on the right
            //No need to template match with left part of the image
            var subTransform = transformed.rowRange(100, 400).colRange(425, 500)
            matchTemplate(subTransform, template3DS, tmpMat, TM_CCOEFF_NORMED)
            val match3ds = Core.minMaxLoc(tmpMat).maxVal

            //Assuming game box is upright, the NDS logo is always on the left
            //No need to template match with right part of the image
            subTransform = transformed.rowRange(100, 400).colRange(1, 75)
            matchTemplate(subTransform, templateDS, tmpMat, TM_CCOEFF_NORMED)
            val matchds = Core.minMaxLoc(tmpMat).maxVal

            runOnUiThread {
                if (match3ds > 0.8)
                    findViewById<TextView>(R.id.textView).text = "3DS GAME"
                else if (matchds > 0.8)
                    findViewById<TextView>(R.id.textView).text = "DS GAME"
                else
                    findViewById<TextView>(R.id.textView).text = ""
            }

            return previewMat
        }
        return Mat()
    }
}