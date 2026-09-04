___

                 
                 CITY-WIDE CAMERA NETWORK
                           │
              ┌────────────┴────────────┐
              │                         │
          Camera 1                  Camera 2 ... Camera N
              │                         │
              └────────────┬────────────┘
                           ↓
                  VIDEO / IMAGE STREAMS
                           ↓
                  ┌─────────────────┐
                  │  Preprocessing  │
                  │ Resize, enhance │
                  │ denoise, etc.   │
                  └────────┬────────┘
                           ↓
                  ┌─────────────────┐
                  │ Vehicle         │
                  │ Detection       │
                  └────────┬────────┘
                           ↓
                  ┌─────────────────┐
                  │ License Plate   │
                  │ Detection       │
                  └────────┬────────┘
                           ↓
                  ┌─────────────────┐
                  │ OCR / ANPR      │
                  │ Plate Reading   │
                  └────────┬────────┘
                           ↓
              PLATE + TIME + CAMERA + LOCATION
                           ↓
                  ┌─────────────────┐
                  │ Data Association│
                  │ / Vehicle       │
                  │ Identification  │
                  └────────┬────────┘
                           ↓
              ┌────────────┴────────────┐
              ↓                         ↓
       Specific Vehicle            All Vehicles
       Tracking                    Aggregated Data
              ↓                         ↓
    Trajectory Reconstruction     Traffic Analytics
              ↓                         ↓
       GIS Route Map              Heatmaps / Density
       + timestamps               Congestion / OD
              │                         │
              └────────────┬────────────┘
                           ↓
                    ALERT ENGINE
                           ↓
              Blacklisted / Suspicious
                    Vehicle Alerts
                           ↓
                    WEB DASHBOARD

Suppose there are three cameras:

```
Camera A                  Camera B                  Camera C
Anna Nagar                Nungambakkam              T. Nagar
    │                          │                         │
    │                          │                         │
    └───────────┐              │              ┌──────────┘
                ↓              ↓              ↓
             VEHICLE ABC 1234 MOVES THROUGH CITY
```

A car passes Camera A.

### Step 1 — Camera captures vehicle

Camera A sends:

```
Image/video
     ↓
Vehicle detected
```

The AI detects:

```
Vehicle = Car
```

Then it detects the number plate:

```
Plate region
     ↓
┌─────────────┐
│ ABC 1234    │
└─────────────┘
```

---

## OCR / ANPR engine

The plate image goes into the OCR model.

```
Plate Image
     ↓
Deep Learning OCR
     ↓
"ABC 1234"
```

The system stores something like:

```
Plate: ABC 1234
Camera: C01
Location: Anna Nagar
Time: 10:32:15
Direction: East
Confidence: 96%
```

The requirement is for the OCR system to achieve **greater than 90% recognition accuracy**, including difficult conditions such as poor lighting, weather, angled plates, motion blur, and damaged plates.

---

# Trajectory tracking

Suppose 10 minutes later the same vehicle reaches Camera B.

Camera B detects:

```
ABC 1234
10:42:31
Camera B
Nungambakkam
```

Then Camera C detects:

```
ABC 1234
10:51:07
Camera C
T. Nagar
```

The system combines these detections:

```
ABC 1234

10:32:15
Camera A
Anna Nagar
       ↓
10:42:31
Camera B
Nungambakkam
       ↓
10:51:07
Camera C
T. Nagar
```

Now you have a **vehicle trajectory**.

On the GIS map it can become:

```
        Camera A
        ●
         \
          \
           ● Camera B
             \
              \
               ● Camera C
```

The problem statement explicitly expects the trajectory engine to reconstruct a vehicle's historical path with **timestamps and camera locations** and display it on a city map.

---

# Then 

The cameras are independent.

Instead:

```
Camera A sees ABC 1234
              ↓
Camera B sees ABC 1234
              ↓
Camera C sees ABC 1234
```

The system says:

> "These observations probably correspond to the same vehicle."

So the architecture needs a **data association / trajectory reconstruction layer**.

Conceptually:

```
Camera detections
       ↓
Plate number
       +
Timestamp
       +
Camera location
       +
Direction
       +
Confidence
       ↓
Vehicle identity
       ↓
Trajectory
```

The plate number is the primary identifier, while timestamps, location and direction help reconstruct a sensible journey.

# Architecture:

```
┌─────────────────────────────────────────────────────────┐
│                  PRESENTATION LAYER                     │
│                                                         │
│  Web Dashboard │ GIS Map │ Search │ Alerts │ Analytics  │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                  APPLICATION LAYER                      │
│                                                         │
│ Trajectory API │ Analytics API │ Alert API │ Query API  │ 
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                     AI LAYER                            │
│                                                         │
│ Vehicle Detection                                       │
│        ↓                                                │
│ License Plate Detection                                 │
│        ↓                                                │
│ OCR / ANPR                                              │
│        ↓                                                │
│ Vehicle Re-identification / Association                 │
│        ↓                                                │
│ Trajectory Reconstruction                               │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                 STREAM PROCESSING                       │
│                                                         │
│ Camera Streams → Message Queue → AI Inference           │
│                              ↓                          │
│                       Event Processing                  │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                    DATA LAYER                           │
│                                                         │
│ Vehicle Events │ Trajectories │ Camera Data             │
│ Traffic Metrics│ Blacklist │ Historical Data            │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│               CAMERA / EDGE LAYER                       │
│                                                         │
│ CCTV │ ANPR Cameras │ Edge Devices │ GPS/Location       │
└─────────────────────────────────────────────────────────┘


```


# camera metadata

```
Camera 1 ──┐
Camera 2 ──┤
Camera 3 ──┤
Camera 4 ──┤──→ City-wide platform
Camera 5 ──┤
...         │
Camera N ──┘
```

Each camera has metadata:

```
Camera ID
Location
Latitude
Longitude
Road
Direction
Camera type
```

For example:

```
{
  "camera_id": "CAM_042",
  "latitude": 13.04,
  "longitude": 80.23,
  "direction": "North"
}
```

# Stream Processing Layer

You don't want every camera to directly dump raw video into your database.

Instead:

```
Camera
   ↓
Video Stream
   ↓
Stream Processor
   ↓
Message Queue
   ↓
AI Inference
```

The message queue acts like a buffer.

For example:

```
Camera 1 ──┐
Camera 2 ──┤
Camera 3 ──┤
Camera 4 ──┤──→ Message Queue
Camera 5 ──┘          │
                      ↓
                 AI Workers
```

This makes the architecture scalable.

If 100 cameras suddenly produce a huge amount of data, the queue prevents the entire system from collapsing.

---

# Traffic Analytics Engine

This works differently.

Trajectory tracking asks:

> **"Where did THIS vehicle go?"**

Traffic analytics asks:

> **"What is happening to ALL vehicles?"**

Suppose the system collects:

```
Vehicle A → A → B → C
Vehicle B → A → B → D
Vehicle C → A → C → D
Vehicle D → B → C → D
```

The analytics engine aggregates this.

It can calculate:

### Traffic density

```
Camera A → 1,200 vehicles/hour
Camera B → 2,500 vehicles/hour
Camera C → 4,100 vehicles/hour
```

### Average speed

```
Road A → 42 km/h
Road B → 18 km/h
Road C → 11 km/h
```

### Origin-Destination

```
             DESTINATION
           A      B      C
        ┌──────────────────
SOURCE A│  -    500    200
       B│ 300     -     800
       C│ 150    700     -
```

This lets authorities understand where vehicles are coming from and where they're going.

The requested system specifically includes **traffic density, origin-destination patterns, congestion bottlenecks, heatmaps, average speeds, route densities and traffic-flow trends**

# Alert Engine

The alert engine sits on top of the AI/data pipeline.

For example, maintain:

```
BLACKLIST

TN01AB1234
TN09XY9876
TN22CD4567
```

When ANPR detects:

```
TN01AB1234
```

the pipeline becomes:

```
Camera
  ↓
ANPR
  ↓
TN01AB1234
  ↓
Blacklist Database
  ↓
MATCH!
  ↓
ALERT
  ↓
Dashboard
```

The system can also detect **route anomalies**.

For example:

```
Normal:
Camera A → B → C

Observed:
Camera A → C → B
```

Depending on the city's road network and timing, this could be flagged for investigation.

The problem statement explicitly requires alerts for blacklisted vehicles and suspicious route anomalies in real time

# Database architecture

You shouldn't put everything into one database.

A sensible architecture would be:

```
                    DATABASES
                        │
        ┌───────────────┼────────────────┐
        ↓               ↓                ↓
   Metadata DB      Event Store      Analytics DB
        │               │                │
   Camera info      Plate events      Aggregations
   Road info        Timestamps        Traffic density
   Locations        Detections        OD matrices
   Configuration    Confidence        Heatmaps
```

### Metadata DB

Stores:

```
Camera ID
Camera location
Road
Direction
Configuration
```

### Event store

Stores individual detections:

```
plate
camera_id
timestamp
confidence
direction
```

### Analytics database

Stores processed information:

```
traffic density
average speed
congestion
OD patterns
route statistics
```

---