import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

import {
  Group,
  Text,
  useMantineTheme,
  Stack,
  Title,
  createStyles,
} from "@mantine/core";
import { Upload, Photo, X } from "tabler-icons-react";
import { Dropzone, MIME_TYPES } from "@mantine/dropzone";

const useStyles = createStyles((theme) => ({
  root: {
    padding: theme.spacing.xl,
    margin: theme.spacing.xl,
  },
  group: { padding: theme.spacing.xl, margin: theme.spacing.xl },
  title: {
    backgroundColor: theme.colors.gray[0],
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
    style={{ minHeight: 220, pointerEvents: "none" }}
  >
    <ImageUploadIcon
      status={status}
      style={{ color: getIconColor(status, theme) }}
      size={80}
    />

    <div>
      <Text size="xl" inline>
        Drag images here or click to select files
      </Text>
      <Text size="sm" color="dimmed" inline mt={7}>
        Attach as many files as you like, each file should not exceed 5mb
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

  const handleImageChange = async (files) => {
    setLoading(true);
    if (files.length === 0) {
      setConfidence(null);
      setPredictedClass(null);
    }

    if (files.length === 1) {
      const imageSrc = await readImageFile(files[0]);
      const image = await createHTMLImageElement(imageSrc);

      // tf.tidy for automatic memory cleanup
      const [predictedClass, confidence] = tf.tidy(() => {
        const tensorImg = tf.browser
          .fromPixels(image)
          .resizeNearestNeighbor([256, 256])
          .toFloat()
          .expandDims();
        const result = model.predict(tensorImg);
        console.log("result");
        console.log(result);
        const predictions = result.dataSync();
        const predicted_index = result.as1D().argMax().dataSync()[0];
        console.log("predictions");
        console.log(predictions);
        console.log("predicted_index");
        console.log(predicted_index);

        const predictedClass = classLabels[predicted_index];
        const confidence = Math.round(predictions[predicted_index] * 100);

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

      setModel(model);
    };

    const getClassLabels = async () => {
      const data = ["canada_ontario_cvo", "none"];

      setClassLabels(data);
    };

    loadModel();
    getClassLabels();
  }, []);

  console.log(loading);
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
        <Dropzone
          loading={loading}
          onDrop={(files) => {
            console.log("accepted files", files);
            handleImageChange(files);
          }}
          onReject={(files) => console.log("rejected files", files)}
          maxSize={3 * 1024 ** 2}
          accept={[MIME_TYPES.png, MIME_TYPES.jpeg]}
        >
          {(status) => dropzoneChildren(status, theme)}
        </Dropzone>
        <Stack
          style={{ marginTop: "2em", width: "12rem" }}
          direction="row"
          spacing={1}
        >
          <Text size="xl">
            {predictedClass === null
              ? "Prediction:"
              : `Prediction: ${predictedClass}`}
          </Text>
          <Text size="xl">
            {confidence === null ? "Confidence:" : `Confidence: ${confidence}%`}
          </Text>
        </Stack>
      </Group>
    </div>
  );
}

export default App;
