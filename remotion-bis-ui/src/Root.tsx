import "./index.css";
import { Composition } from "remotion";
import { BisPlatformWalkthrough } from "./Composition";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="BISPlatformWalkthrough"
        component={BisPlatformWalkthrough}
        durationInFrames={360}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
