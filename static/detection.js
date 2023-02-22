const $ = (selector)=>{
    return document.querySelector(selector);
}

const videoElement = $('.input_video');
const canvasElement = $('.output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const outputCanv = $('#output');
const ouputText = $('#result');
let sequence = []

function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);


    // Only overwrite existing pixels.
    canvasCtx.globalCompositeOperation = 'source-in';
    canvasCtx.fillStyle = '#00FF00';
    canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);

    // Only overwrite missing pixels.
    canvasCtx.globalCompositeOperation = 'destination-atop';
    canvasCtx.drawImage(
        results.image, 0, 0, canvasElement.width, canvasElement.height);

    canvasCtx.globalCompositeOperation = 'source-over';
    drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS,
        { color: '#00FF00', lineWidth: 1 });
    drawLandmarks(canvasCtx, results.poseLandmarks,
        { color: '#FF0000', lineWidth: 2 });
    drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_CONTOURS,
        { color: '#C0C0C070', lineWidth: 1 });
    drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS,
        { color: '#CC0000', lineWidth: 5 });
    drawLandmarks(canvasCtx, results.leftHandLandmarks,
        { color: '#00FF00', lineWidth: 2 });
    drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS,
        { color: '#00CC00', lineWidth: 5 });
    drawLandmarks(canvasCtx, results.rightHandLandmarks,
        { color: '#FF0000', lineWidth: 2 });
    canvasCtx.restore();
    // gets landmark keypoints and fill zeros array if landmarks not visible
    let pose = results.poseLandmarks ?
        Array.from(results.poseLandmarks, (res) => [res.x, res.y, res.z, res.visibility]).flat() :
        Array(33 * 4).fill(0);
    let face = results.faceLandmarks ?
        Array.from(results.faceLandmarks, (res) => [res.x, res.y, res.z]).flat() :
        Array(468 * 3).fill(0);
     if(face.length!=1404){
        face.splice(face.length - 30, 30);
    }
    let lh = results.leftHandLandmarks ?
        Array.from(results.leftHandLandmarks, (res) => [res.x, res.y, res.z]).flat() :
        Array(21 * 3).fill(0);

    let rh = results.rightHandLandmarks ?
        Array.from(results.rightHandLandmarks, (res) => [res.x, res.y, res.z]).flat() :
        Array(21 * 3).fill(0);

    let combined_arrs = [].concat(pose, face, lh, rh).flat();
    sequence.push(combined_arrs);
}

const holistic = new Holistic({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
    }
});
holistic.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: true,
    smoothSegmentation: true,
    refineFaceLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});
let frame_count = 0;
const camera = new Camera(videoElement, {
    onFrame: async () => {
        await holistic.send({ image: videoElement });
        holistic.onResults(onResults);
        if (frame_count ==  30) {
        console.log(sequence)
        let result = "";
            frame_count = 0;
            fetch('/test', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ data: sequence })
            })
                .then(response => response.json())
                .then(data => ouputText.textContent=data.result)
                .catch(error => console.error(error));
            sequence = []
        }
         frame_count++;
    },
    width: 1280,
    height: 720
});
camera.start();