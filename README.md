# VisualOdometry
![image](https://github.com/IldarGreat/VisualOdometry/assets/90307025/dca96aed-b714-4d80-ad25-337e0bc28656)


<h2>Config yaml file</h2>
The config file (config.yaml) is located in the root of the project and allows you to configure the basic parameters of odometry, camera and so on <br>
Each possible value of each variable in the config will be described below <br>
<h3>camera</h3>
<ul>
  <li>fx</li>
  <li>fy</li>
  <li>cx</li>
  <li>cy</li>
</ul>
<h3>odometry</h3>
<ul>
  <li>detector</li>
    <ul>
      <li>ORB</li>
      <li>SIFT</li>
      <li>AKAZE</li>
      <li>BRISK</li>
      <li>FAST</li>
      <li>BLOB</li>
      <li>SURF</li>
    </ul>
  <li>descriptor</li>
  <ul>
      <li>ORB</li>
      <li>SIFT</li>
      <li>AKAZE</li>
      <li>BRISK</li>
      <li>SURF</li>
    </ul>
  <li>matcher</li>
  <ul>
      <li>BFMatcher</li>
      <li>FLANN</li>
    </ul>
  <li>opticalFlow</li>
  <ul>
      <li>None</li>
      <li>Farneback</li>
      <li>PyrLK</li>
    </ul>
  <li>methodForComputingE</li>
  <ul>
      <li>RANSAC</li>
      <li>LMEDS</li>
    </ul>
</ul>
<h3>data</h3>
<ul>
  <li>liveCamera</li>
   <ul>
      <li>'None' if you're not using a live camera.</li>
      <li>Or specify the port of the connected camera</li>
    </ul>
  <li>video</li>
   <ul>
      <li>'None' if you're not using a video.</li>
      <li>Or specify the path to the video</li>
    </ul>
  <li>images</li>
   <ul>
      <li>'None' if you're not using a images.</li>
      <li>Or specify the path to the images</li>
    </ul>
  <li>groundTruth</li>
   <ul>
      <li>'None' if you're not using a ground truth - bad for monocular vo.</li>
      <li>Or specify the path of the ground truth poses</li>
    </ul>
  <li>groundTruthTemplate</li>
   <ul>
      <li>'None' if you're not using a ground truth.</li>
      <li>KITTI</li>
     <li>Default</li>
    </ul>
</ul>
<h3>settings</h3>
<ul>
  <li>logs</li>
    <ul>
      <li>None</li>
      <li>True</li>
    </ul>
  <li>seeTracking</li>
  <ul>
      <li>None</li>
      <li>True</li>
    </ul>
</ul>

@By Ildar Idiyatov, Samara University
