
import { useRef, useState, useEffect, useCallback} from 'react';
import axios from 'axios';
import { useMutation } from 'react-query';
import Webcam from 'react-webcam';

import useInterval from './useInterval';

// material-UI
import { Grid, Box } from '@material-ui/core/';
import { red, pink, purple, deepPurple, indigo, blue, 
          lightBlue, cyan, teal, green, lightGreen, lime,
          yellow, amber, orange, deepOrange, } from '@material-ui/core/colors/';

// d3
import { select } from "d3";

interface maskData {
    check : boolean
    bboxes : number[][],
    labels : string[],
    segmentations : string[],
}
interface picture {
  image : string;
};

interface toggles{
  interval : number | number[],
  classification : boolean,
  segmentation : boolean,
  detection : boolean
};

// 16 bright colors
const colorMap : any = [ 
  red, pink, purple, deepPurple, indigo, blue, lightBlue, 
  cyan, teal, green, lightGreen, lime, yellow, amber, orange,
  deepOrange];

const initData : maskData ={
  'check' : true,
  'bboxes' : [[0, 0, 0, 0]],
  'labels' : [''],
  'segmentations' : ['0,0']
}

// 10 colors
const initColors :string[] = [
  red[400], purple[400], indigo[400], blue[400], cyan[400], 
  green[400], lime[400], yellow[400], amber[400], deepOrange[400]
]

const videoConstraints = {
    width: 448,
    height: 448,
    facingMode: "user"
};

const WebcamDrawing = (props : toggles) => {

    const svgRef = useRef(null);
    const webcamRef = useRef<any>(null);
    const [previewURL, setPreview] = useState<string>('');
    const [colors, setColors] = useState<string[]>(initColors) // max colors : 10

    const [receivedMaskData, setMaskData] = useState<maskData>(initData);
    const [labelData, setLabelData] = useState<string[]>(initData.labels);
    const [bboxData, setBboxData] = useState<number[][]>(initData.bboxes);
    const [segmentation, setSegmentation] = useState<string[]>(initData.segmentations);

    // boostcamp
    // const api = axios.create({
    //   baseURL: `http://49.50.165.199:6010/`,
    // })

    // AWS
    const api = axios.create({
      baseURL: `http://54.180.91.142/`,
    })
  
    const [mutateCreate] = useMutation(
        (data: picture) => api.post('masks', data), { 
      onSuccess: (res) => {
        const result = JSON.parse(res.data)
        if(result['check']){ // success
          setMaskData(result);
        } 
        else { colorChange(); } // success without results
      },
      onError : (e) => {
        console.log(e);
      }
    })

    // set new 10 colors! 
    const colorChange = useCallback(() => {
      const newColors : string[] = [];
      for (let i = 0; i < 10; i++){
        newColors.push(colorMap[Math.floor(Math.random() * 14)][(Math.floor(Math.random() * 8) + 1)*100])
      }
      setColors(newColors)
    },[setColors]);

    const capture = useCallback(() => {
      const image : string = webcamRef.current.getScreenshot();
      setPreview(image) // base64 image file
    }, [webcamRef, setPreview]);

    const submit = () => {
      const json : string = JSON.stringify({
        data : previewURL,
      })
      const obj : picture = JSON.parse(json);
      if(previewURL !== '') mutateCreate(obj);
    }

    // update Received Mask Data
    useEffect(() => {
      setLabelData(receivedMaskData.labels);
      setBboxData(receivedMaskData.bboxes);
      setSegmentation(receivedMaskData.segmentations);
    }, [receivedMaskData, setLabelData, setBboxData, setSegmentation])

    // drawing classification labels
    useEffect(() => {
      if(svgRef.current && props.classification){
      const svg = select(svgRef.current);
      
      svg
        .selectAll('text')
        .data(bboxData)
        .join(
          (enter) => enter.append('text'),
          (update) => update.attr("class", "updated"),
          (exit) => exit.remove()
        )
        .attr('x', (data) => (data[0] - 15))
        .attr('y', (data) => (data[1] - 5))
        .data(labelData!)
        .text(data => data)
        .attr('font-family', 'Arial')
        .attr('font-size', "11px")
        .attr('text-align', 'center')
        .attr('fill', 'red')
        .attr('stroke-width', 1)
    } else select("text").remove();
    }, [bboxData, labelData, props.classification]);

    // drawing bounding box
    useEffect(() => {
      if(svgRef.current && props.detection){
      const svg = select(svgRef.current);
      
      svg
        .selectAll("rect")
        .data(bboxData)
        .join(
          (enter) => enter.append("rect"),
          (update) => update.attr("class", "updated"),
          (exit) => exit.remove()
        )
        .attr('width', (data : number[]) => data[2] - data[0])
        .attr('height', (data : number[]) => data[3] - data[1])
        .attr('x', (data : number[]) => data[0])
        .attr('y', (data : number[]) => data[1])
        .attr('fill', 'transparent')
        .attr('stroke', 'red')
        .attr('stroke-width', 2)
      } else select("rect").remove();
    }, [bboxData, props.detection]);
    
    // drawing segmentation mask
    useEffect(() => {
      if(svgRef.current && props.segmentation){

      const svg = select(svgRef.current);
      const colorNumbers = [0,1,2,3,4,5,6,7,8,9];

      svg
        .selectAll("polygon")
        .data(segmentation)
        .join(
          (enter) => enter.append("polygon"),
          (update) => update.attr("class", "updated"),
          (exit) => exit.remove()
        )
        .attr('points', (data : string) => data)
        .data(colorNumbers)
        .attr('fill', (data : number) => colors[data])
        .attr('stroke-width', 2)
        .attr('opacity', 0.5)
      } else select('polygon').remove();
    }, [segmentation, colors, props.segmentation]);

    // set fps
    useInterval (() =>{
      if(webcamRef.current) {
        try{
          capture();
          submit();
        }
        catch (e) {
          console.log(e);
        }
      }
    }, (1000 / (isNaN(Number(props.interval))? 1 : Number(props.interval))));

    // useEffect clean-up
    useEffect(() => {
      return () => setPreview('');
    }, []);

    return (
        <>
          <Grid container justify="center" alignItems="center">
          <Webcam
              audio={false}
              height={window.innerHeight > 640 ? 448 : 224}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              width={window.innerWidth > 640 ? 448 : 224}
              videoConstraints={videoConstraints}
          />
          <svg ref={svgRef}
            style ={{
              position: "absolute",
              marginLeft: "auto",
              marginRight: "auto",
              textAlign: "center",
              zIndex: 9,
              width: window.innerWidth > 640 ? 448 : 224,
              height: window.innerHeight > 640 ? 448 : 224, 
            }}
          />
          </Grid>
          <Box mt = "1rem"/>
      </>
  );
}

export default WebcamDrawing