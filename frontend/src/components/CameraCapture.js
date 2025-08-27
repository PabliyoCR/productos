import React, { useRef, useEffect, useState } from 'react';

const CameraCapture = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const capturedImageRef = useRef(null); // Ref for captured image div
  const resultRef = useRef(null); // Ref for result div
  const [capturedImage, setCapturedImage] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    const enableStream = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing the camera: ", err);
      }
    };

    enableStream();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    if (capturedImage && capturedImageRef.current) {
      capturedImageRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [capturedImage]);

  useEffect(() => {
    if (result && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [result]);

  const takePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const context = canvasRef.current.getContext('2d');
      canvasRef.current.width = videoRef.current.videoWidth;
      canvasRef.current.height = videoRef.current.videoHeight;
      context.drawImage(videoRef.current, 0, 0, videoRef.current.videoWidth, videoRef.current.videoHeight);
      setCapturedImage(canvasRef.current.toDataURL('image/png'));
    }
  };

  const sendPhoto = async () => {
    if (!capturedImage) {
      alert("Primero, toma una foto.");
      return;
    }

    try {
      const response = await fetch(capturedImage);
      const blob = await response.blob();
      const file = new File([blob], "captured-image.png", { type: "image/png" });

      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("http://localhost:8000/buscar-producto", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error("Error sending the image: ", err);
      alert("Error al enviar la imagen.");
    }
  };

  return (
    <div>
      <h2>Captura de Cámara</h2>
      <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', maxWidth: '500px' }} />
      <button onClick={takePhoto}>Tomar Foto</button>
      {capturedImage && (
        <div ref={capturedImageRef} style={{ marginBottom: '48px' }}> {/* Apply ref here */}
          <h3>Foto Capturada:</h3>
          <img src={capturedImage} alt="Captured" style={{ width: '100%', maxWidth: '500px' }} />
          <button onClick={sendPhoto}>Enviar Foto</button>
        </div>
      )}
      <canvas ref={canvasRef} style={{ display: 'none'}} />

      {result && (
        <div ref={resultRef}> {/* Apply ref here */}
          <h3>Resultados del Servidor:</h3>
          {result.resultados && result.resultados.length > 0 ? (
            <div>
              {result.resultados.map((item, index) => (
                <div key={index} style={{ marginBottom: '20px', border: '1px solid #ccc', padding: '10px' }}>
                  <h3>{item.familia} &gt; {item.categoria} &gt; {item.subcategoria}</h3>
                  <img src={item.imageurl} alt={item.producto} style={{ width: '100%', maxWidth: '300px', height: 'auto' }} />
                  <p>{item.producto}</p>
                </div>
              ))}
            </div>
          ) : (
            <p>No se encontraron resultados.</p>
          )}
        </div>
      )}
    </div>
  );
};

export default CameraCapture;
