// Copyright Naoki Shibata 2018. Distributed under the MIT License.

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <time.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#include <ctime>

#include "vec234.h"

#include "helper.h"
#include "oclhelper.h"

#include "oclimgutil.h"
#include "oclpolyline.h"
#include "oclrect.h"

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/Image.h>

#include <iostream>
#include <thread>

using namespace std;
Mat buffer_frame;
uint64_t fc = 0;
uint64_t cfc = -1;
bool cp = false;

void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
  cout << "frame received\n";
  auto i = cv_bridge::toCvCopy(msg);
  while (cp)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  buffer_frame = (*i).image;
  cvtColor(buffer_frame, buffer_frame, COLOR_RGB2BGR);
  fc++;
}

static void showRect(rect_t rect, int r, int g, int b, int thickness, Mat &img)
{
  for (int i = 0; i < 4; i++)
  {
    line(img, cvPoint(rect.c2[i].a[0], rect.c2[i].a[1]), cvPoint(rect.c2[(i + 1) % 4].a[0], rect.c2[(i + 1) % 4].a[1]), Scalar(r, g, b), thickness, 8, 0);
  }

  line(img,
       cvPoint(rect.c2[0].a[0], rect.c2[0].a[1]),
       cvPoint(rect.c2[2].a[0], rect.c2[2].a[1]), Scalar(r, g, b), 1, 8, 0);

  line(img,
       cvPoint(rect.c2[1].a[0], rect.c2[1].a[1]),
       cvPoint(rect.c2[3].a[0], rect.c2[3].a[1]), Scalar(r, g, b), 1, 8, 0);
}

static int fourcc(const char *s)
{
  return (((uint32_t)s[0]) << 0) | (((uint32_t)s[1]) << 8) | (((uint32_t)s[2]) << 16) | (((uint32_t)s[3]) << 24);
}

Mat get_new_frame()
{
  while (fc == cfc || fc == 0)
  {
    ros::spinOnce();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  cp = true;
  cfc = fc;
  Mat rv;
  buffer_frame.copyTo(rv);
  cp = false;
  return rv;
}

double angle(Point2d a, Point2d b, Point2d c)
{
  auto aa = abs(acos(abs((b - a).dot(c - b) / sqrt((b - a).dot(b - a) * (c - b).dot(c - b)))) - asin(1));
  return aa;
}

int main(int argc, char **argv)
{

  ros::init(argc, argv, "precision_landing");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("/bebop/image_raw", 1, imageCallback);

  //
  get_new_frame();
  int iw = buffer_frame.cols;
  int ih = buffer_frame.rows;

  printf("Resolution : %d x %d\n", iw, ih);

  //

  VideoWriter *writer = NULL;
  const char *winname = "Rectangle Detection Demo";

  double aov = 72;

  int did = 0;
  cvNamedWindow(winname, WINDOW_NORMAL);
  printf("Horizontal angle of view : %g degrees\n", aov);

  cl_device_id device = simpleGetDevice(did);
  printf("%s\n", getDeviceName(device));
  cl_context context = simpleCreateContext(device);

  cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

  if (loadPlan("plan.txt", device) != 0)
    printf("No plan\n");

  oclimgutil_t *oclimgutil = init_oclimgutil(device, context);
  oclpolyline_t *oclpolyline = init_oclpolyline(device, context);
  oclrect_t *oclrect = init_oclrect(oclimgutil, oclpolyline, device, context, queue, iw, ih);

  //

  const double tanAOV = tan(aov / 2 / 180.0 * M_PI);
  Mat vimg, img[2];

  int nFrame = 0, lastNFrame = 0;
  uint64_t tm = currentTimeMillis();
  clock_t begin = clock();

  for (;;)
  {
    //if (!cap->retrieve(vimg, 0)) break;
    get_new_frame();
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    nFrame++;

    buffer_frame.copyTo(img[nFrame & 1]);

    uint8_t *data = (uint8_t *)img[nFrame & 1].data;
    int ws = img[nFrame & 1].step;

    oclrect_enqueueTask(oclrect, data, ws);

    rect_t *ret = oclrect_pollTask(oclrect, tanAOV);
    Mat i2;
    cvtColor(img[nFrame & 1], i2, COLOR_BGR2RGB);
    for (int i = 1; i < ret->nItems; i++)
    {
      auto p1 = Point2d(ret[i].c2[0].a[0], ret[i].c2[0].a[1]);
      auto p2 = Point2d(ret[i].c2[1].a[0], ret[i].c2[1].a[1]);
      auto p4 = Point2d(ret[i].c2[3].a[0], ret[i].c2[3].a[1]);
      auto p3 = Point2d(ret[i].c2[2].a[0], ret[i].c2[2].a[1]);

      auto d1 = (p3 - p1) / sqrt((p3 - p1).dot(p3 - p1));
      auto d2 = (p4 - p2) / sqrt((p4 - p2).dot(p4 - p2));

      auto c1 = i2.at<Vec3b>(p1 + d1 * 5);
      auto c2 = i2.at<Vec3b>(p2 + d2 * 5);
      auto c3 = i2.at<Vec3b>(p3 - d1 * 5);
      auto c4 = i2.at<Vec3b>(p2 - d2 * 5);
      int cc1 = ((int)c1[1] + c2[1] + c3[1] + c4[1]) / 4;
      int cc0 = ((int)c1[0] + c2[0] + c3[0] + c4[0]) / 4;
      int cc2 = ((int)c1[2] + c2[2] + c3[2] + c4[2]) / 4;
      double dd = angle(p1, p2, p3) + angle(p2, p3, p4) + angle(p3, p4, p1) + angle(p4, p1, p2);

      if (dd < 0.4)
      {
        dd = abs(atan(d1.y / d1.x));
        std::ostringstream strs;
        strs << dd;
        std::string str = strs.str();

        if (dd < 0.4)
        {
          putText(img[nFrame & 1], str, p1,
                  FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255, 0, 0), 1, CV_AA);
          showRect(ret[i], cc2, cc1, cc0, 2, img[nFrame & 1]);
        }
      }
      double fps = (double)nFrame / elapsed_secs;
      std::ostringstream strs2;
      strs2 << fps;
      std::string str2 = strs2.str();

      putText(img[nFrame & 1], str2, cvPoint(10, 10),
              FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, CV_AA);
      if(nFrame==100)
      {
        begin = clock();
        nFrame=0;
      }
      imshow(winname, img[nFrame & 1]);
      int key = waitKey(1) & 0xff;
      if (key == 27 || key == 13)
        break;
    }
  }

  //

  dispose_oclrect(oclrect);
  dispose_oclpolyline(oclpolyline);
  dispose_oclimgutil(oclimgutil);

  //

  ce(clReleaseCommandQueue(queue));
  ce(clReleaseContext(context));

  //

  if (writer != NULL)
    delete writer;
  destroyAllWindows();

  //

  exit(0);
}
