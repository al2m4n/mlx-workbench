import { Navigate, Route, Routes } from "react-router-dom";

import Layout from "./components/Layout";
import ChatPage from "./pages/Chat";
import ModelsPage from "./pages/Models";
import MetricsPage from "./pages/Metrics";
import AudioPage from "./pages/Audio";
import VideoPage from "./pages/Video";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Navigate to="/chat" replace />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/metrics" element={<MetricsPage />} />
        <Route path="/audio" element={<AudioPage />} />
        <Route path="/video" element={<VideoPage />} />
      </Route>
    </Routes>
  );
}
