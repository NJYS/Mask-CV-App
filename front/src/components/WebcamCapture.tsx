import { useRef, useCallback} from 'react';
import Webcam from 'react-webcam';
import Button from '@material-ui/core/Button';
import Grid from '@material-ui/core/Grid';
import Box from '@material-ui/core/Box';

const videoConstraints = {
    width: 448,
    height: 448,
    facingMode: "user"
  };

const WebcamCapture = (props : any) => {
    const webcamRef : React.MutableRefObject<any> = useRef<any>(null);
    const {camToggle, setPreview} = props;

    const capture = useCallback(() => {
        const imageSrc = webcamRef.current.getScreenshot();
        setPreview(imageSrc) // base64 image file
        camToggle();
      }, [webcamRef, camToggle, setPreview]);

    return (
          <>
            <Grid container justify="center" alignItems="center">
              <Webcam
                audio={false}
                height={window.innerHeight> 512? 448 : 256}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                width={window.innerHeight > 512? 448 : 256}
                videoConstraints={videoConstraints}
              />
            </Grid>
            <Box mt = "1rem"/>
            <Grid container spacing={5} direction="row" alignItems="center" justify="center">
              <Grid item>
                <Button variant="contained" size="small" color="primary" onClick={capture}>캡쳐</Button>
              </Grid>
              <Grid item>
                <Button variant="contained" size="small" color="primary" onClick={camToggle}>종료</Button>
              </Grid>
            </Grid>
        </>
    );
}

export default WebcamCapture
