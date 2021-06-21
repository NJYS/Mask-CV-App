import { useState } from 'react';

// components
import WebcamDrawing from './components/WebcamDrawing';

// material-UI
import Box from '@material-ui/core/Box';
import Button from '@material-ui/core/Button';
import Grid from '@material-ui/core/Grid';
import Container from '@material-ui/core/Container';
import IconButton from '@material-ui/core/IconButton';
import { withStyles, makeStyles} from '@material-ui/core/styles';
import Tooltip from '@material-ui/core/Tooltip';
import Paper from '@material-ui/core/Paper';
import VideocamIcon from '@material-ui/icons/Videocam';
import VideocamOffIcon from '@material-ui/icons/VideocamOff';
import Slider from '@material-ui/core/Slider';
import Typography from '@material-ui/core/Typography';

const PrettoSlider = withStyles({
  root: {
    color: '#52af77',
    height: 8,
  },
  thumb: {
    height: 24,
    width: 24,
    backgroundColor: '#fff',
    border: '2px solid currentColor',
    marginTop: -8,
    marginLeft: -12,
    '&:focus, &:hover, &$active': {
      boxShadow: 'inherit',
    },
  },
  active: {},
  valueLabel: {
    left: 'calc(-50% + 4px)',
  },
  track: {
    height: 8,
    borderRadius: 4,
  },
  rail: {
    height: 8,
    borderRadius: 4,
  },
})(Slider);

const useStyles = makeStyles((theme) => ({
  root : {
    width : 500,
  },
  button: {
    '& > *': {
      margin: theme.spacing(1),
    },
  },
  input: {
    display: 'none',
  },
  main : {
      padding : '2em',
      backgroundColor: '#757de8',
      height: '60vh', 
      overflow: 'auto',
  },
  progressCircle : {
    margin:'1em'
  }
}));

function RealTime(){
    const classes = useStyles();

    const [camState, setCam] = useState<boolean>(false);
    const [classState, setClass] = useState<boolean>(true);
    const [segState, setSeg] = useState<boolean>(true);
    const [detState, setDet] = useState<boolean>(true);
    const [interval, setInterval] = useState<number | number[]>(7);
    
    // webcam
    const camToggle = () => {
      setCam(camState => !camState);
    }

    // toggle buttons
    const classToggle = () => {
      setClass(classState => !classState);
    }

    const segToggle = () => {
      setSeg(segState => !segState);
    }

    const detToggle = () => {
      setDet(detState => !detState);
    }

    return (
      <>
      <Grid container spacing={0} direction="column" alignItems="center" justify="center">
        <Box mt="10rem"/>
        <Container maxWidth="sm">
          <Grid item>
          <Paper elevation={4} className ={classes.main}>
            {camState ? 
              <WebcamDrawing 
                interval = {interval}
                classification = {classState} 
                segmentation = {segState}
                detection = {detState} /> 
              : null
            }
            <Box mt="3rem"/>
          </Paper>
          </Grid>
        </Container>
        <Box mb="0.5rem"/>
        <Grid container spacing={3} direction="row" alignItems="center" justify="center">
          <Grid item>
            <Tooltip title="classification" arrow>
              {classState? <Button variant="contained" size="small" color="secondary" onClick = {classToggle}>class🧡</Button>
                : <Button variant="contained" size="small" color="primary" onClick = {classToggle}>class🧡</Button>
              }
            </Tooltip>
          </Grid>
          <Grid item>
            <Tooltip title="segmentation" arrow>
              {segState? <Button variant="contained" size="small" color="secondary" onClick = {segToggle}>seg💛</Button>
                : <Button variant="contained" size="small" color="primary" onClick = {segToggle}>seg💛</Button>
              }
            </Tooltip>
          </Grid>
          <Grid item>
          <Tooltip title="detection" arrow>
              {detState ? <Button variant="contained" size="small" color="secondary" onClick = {detToggle}>det💙</Button>
                : <Button variant="contained" size="small" color="primary" onClick = {detToggle}>det💙</Button>
              }
            </Tooltip>
          </Grid>
          <Grid item>
            <label htmlFor="open_webcam">
            <Tooltip title="webcam on/off" arrow>
            <IconButton color="primary" aria-label="open_webcam" component="span" onClick={camToggle}>
                {camState ? <VideocamOffIcon fontSize = 'large'/> : <VideocamIcon fontSize = 'large'/>}
            </IconButton>
            </Tooltip>
            </label>
          </Grid>
          <Grid item>
            <Typography gutterBottom>　　fps 설정　　</Typography>
            <PrettoSlider valueLabelDisplay="auto" aria-label="pretto slider" 
              defaultValue={5} min = {1} max = {10} step = {1} 
              onChange = {(e, value : number | number[]) => {setInterval(value)}}
            />
          </Grid>
        </Grid>
      </Grid>
      <Box mb="0.5rem"/>
    </> 
    );
}

export default RealTime;