import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import { createWorker } from "tesseract.js";

import {
  Group,
  Text,
  useMantineTheme,
  Title,
  createStyles,
} from "@mantine/core";
import { Upload, Photo, X } from "tabler-icons-react";
import { Dropzone, MIME_TYPES } from "@mantine/dropzone";

const useStyles = createStyles((theme) => ({
  root: {
    padding: theme.spacing.md,
    margin: theme.spacing.md,
  },
  group: { padding: theme.spacing.md, margin: theme.spacing.md },
  title: {
    backgroundColor: theme.colors.gray[0],
  },
  img: {
    maxWidth: 200,
    maxHeight: 150,
  },
}));

function getIconColor(status, theme) {
  return status.accepted
    ? theme.colors.green[6]
    : status.rejected
    ? theme.colors.red[6]
    : theme.colors.gray[7];
}

function ImageUploadIcon({ status, ...props }) {
  if (status.accepted) {
    return <Upload {...props} />;
  }

  if (status.rejected) {
    return <X {...props} />;
  }

  return <Photo {...props} />;
}

export const dropzoneChildren = (status, theme) => (
  <Group
    position="center"
    spacing="xl"
    style={{ minHeight: 120, pointerEvents: "none" }}
  >
    <ImageUploadIcon
      status={status}
      style={{ color: getIconColor(status, theme) }}
      size={80}
    />

    <div>
      <Text size="xl" inline>
        Drag an image here or click to select a file
      </Text>
      <Text size="sm" color="dimmed" inline mt={7}>
        File should not exceed 5mb, jpeg or png only.
      </Text>
    </div>
  </Group>
);

const readImageFile = (file) => {
  return new Promise((resolve) => {
    const reader = new FileReader();

    reader.onload = () => resolve(reader.result);

    reader.readAsDataURL(file);
  });
};

const createHTMLImageElement = (imageSrc) => {
  return new Promise((resolve) => {
    const img = new Image();

    img.onload = () => resolve(img);

    img.src = imageSrc;
  });
};

function App() {
  const theme = useMantineTheme();
  const { classes } = useStyles();

  const [loading, setLoading] = useState(false);
  const [confidence, setConfidence] = useState(null);
  const [predictedClass, setPredictedClass] = useState(null);

  const [model, setModel] = useState(null);
  const [classLabels, setClassLabels] = useState(null);

  const [picture, setPicture] = useState(null);
  const [picturePath, setPicturePath] = useState(null);
  const [errorText, setErrorText] = useState(null);

  // OCR part:
  const [ocr, setOcr] = useState("");
  const [clinicName, setClinicName] = useState("");
  const [clinicAddress, setClinicAddress] = useState("");
  const [progress, setProgress] = useState(0);

  const [workerStatus, setWorkerStatus] = useState(null);
  const [worker, setWorker] = useState(
    createWorker({
      logger: (m) => {
        console.log(m);
        setWorkerStatus(m.status);
        setProgress(parseInt(m.progress * 100));
      },
    })
  );

  useEffect(() => {
    const getTextFromImage = async () => {
      if (!picture || predictedClass === "none") return;
      await worker.load();
      await worker.loadLanguage("eng");
      await worker.initialize("eng");
      const {
        data: { text },
      } = await worker.recognize(picture);
      console.log(`OCR text: ${text}`);

      if (predictedClass === "canada_ontario_cvo") {
        // look up position of clinic name and adress

        let start = null;
        start = text.search("certifies that:\n") + 15;
        if (!start) start = text.search("certifies that\n") + 14;
        const end = text.search("\nhaving been duly inspected");
        const array = text.substring(start, end).split(/\r?\n/);
        console.log("start");
        console.log(start);
        console.log("end");
        console.log(end);
        console.log(array);
        setClinicName(
          array[0] === " " || array[0] === "" ? array[1] : array[0]
        );
        setClinicAddress(
          array[0] === " " || array[0] === "" ? array[2] : array[1]
        );
        setOcr(text);
      }
    };

    getTextFromImage();
  }, [picture, classLabels, predictedClass, worker]);

  const handleImageChange = async (files) => {
    setLoading(true);
    if (files.length === 0) {
      setConfidence(null);
      setPredictedClass(null);
    }

    if (files.length === 1) {
      const imageSrc = await readImageFile(files[0]);
      const image = await createHTMLImageElement(imageSrc);

      setPicture(URL.createObjectURL(files[0]));
      setPicturePath(files[0].path);
      // tf.tidy for automatic memory cleanup
      const [predictedClass, confidence] = tf.tidy(() => {
        const tensorImg = tf.browser
          .fromPixels(image)
          .resizeNearestNeighbor([256, 256])
          .toFloat()
          .expandDims(0);
        // console.log("tensorImg");
        // console.log(tensorImg);
        const result = model.predict(tensorImg);
        // result.print();

        // console.log("result");
        // console.log(result);
        const predictions = result.dataSync();
        const predicted_index = result.as1D().argMax().dataSync()[0];
        // console.log("predictions");
        // console.log(predictions);
        // console.log("predicted_index");
        // console.log(predicted_index);
        // const squaredpredictions = predictions.map((p) => p * p);
        // console.log("squaredpredictions");
        // console.log(squaredpredictions);;
        const predictionsTensor = tf.tensor1d(predictions);
        const score = predictionsTensor.softmax().max().dataSync()[0];

        // console.log("score");
        // console.log(score);
        const predictedClass = classLabels[predicted_index];
        const confidence = Math.round(score * 100);
        return [predictedClass, confidence];
      });

      setPredictedClass(predictedClass);
      setConfidence(confidence);
    }
    setLoading(false);
  };

  useEffect(() => {
    const loadModel = async () => {
      const model_url = "model/model.json";

      const model = await tf.loadGraphModel(model_url);
      // console.log("model");
      // console.log(model);

      setModel(model);
    };

    const getClassLabels = async () => {
      const data = ["canada_ontario_cvo", "none"];

      setClassLabels(data);
    };

    loadModel();
    getClassLabels();
  }, []);

  return (
    <div className={classes.root}>
      <Title order={3} align="center">
        CSI4106 Project: Veterinary Certificate of Accreditation Classifier
      </Title>
      <Group
        position="left"
        spacing="xs"
        direction="column"
        align="center"
        className={classes.group}
      >
        <Text size="xl">Firas Jribi</Text>
        <Text size="xl">Emilie Fortin</Text>
        <Text size="xl">Abir Boutahri</Text>
      </Group>

      <Group position="center" spacing="xl">
        {model ? (
          <div>
            <Dropzone
              loading={loading}
              onDrop={(files) => {
                console.log("accepted files", files);
                setErrorText(null);
                handleImageChange(files);
              }}
              onReject={(files) => {
                setErrorText(
                  `File ${files[0].file.name} rejected: ${files[0].errors[0].message}`
                );
                console.log("rejected files", files);
                setConfidence(null);
                setPredictedClass(null);
                setPicture(null);
              }}
              maxSize={3 * 1024 ** 2}
              accept={[MIME_TYPES.png, MIME_TYPES.jpeg]}
            >
              {(status) => dropzoneChildren(status, theme)}
            </Dropzone>
            <Group className={classes.group} direction="row" spacing={1}>
              <img
                className={classes.img}
                alt=""
                src={picture && picture}
              ></img>
              <Group className={classes.group} direction="column" spacing={1}>
                {!errorText && (
                  <>
                    <Text size="xl">
                      <b>Image:</b> {` ${picturePath ?? ""}`}
                    </Text>
                    <Text size="xl">
                      <b>Prediction:</b> {` ${predictedClass ?? ""}`}
                    </Text>
                    <Text size="xl">
                      <b>Confidence:</b> {` ${confidence ?? ""}%`}
                    </Text>
                    {workerStatus === "recognizing text" && progress !== 100 ? (
                      <div>
                        <Text size="lg">
                          (Analysing Text: {` ${progress ?? ""}%`})
                        </Text>
                      </div>
                    ) : progress === 100 ? (
                      <div>
                        <Text size="xl">
                          <b>Clinic Name:</b> {` ${clinicName ?? ""}`}
                        </Text>
                        <Text size="xl">
                          <b>Clinic Address:</b> {` ${clinicAddress ?? ""}`}
                        </Text>
                      </div>
                    ) : (
                      <></>
                    )}
                  </>
                )}

                {errorText && (
                  <Text size="xl">
                    <b>ERROR:</b>
                    {` ${errorText}`}
                  </Text>
                )}
              </Group>
            </Group>
          </div>
        ) : (
          <div>Loading model...</div>
        )}
      </Group>
    </div>
  );
}

export default App;
