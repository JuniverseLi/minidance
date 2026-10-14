import React from "react";
import { Composition, Still } from "remotion";
import { ArchDiT } from "./ArchDiT";
import { ArchPipeline } from "./ArchPipeline";
import { DEMO_FRAMES, Demo } from "./Demo";

export const Root: React.FC = () => (
  <>
    <Composition id="Demo" component={Demo} durationInFrames={DEMO_FRAMES} fps={30} width={1920} height={1080} />
    <Still id="ArchPipeline" component={ArchPipeline} width={1840} height={1040} />
    <Still id="ArchDiT" component={ArchDiT} width={1640} height={1120} />
  </>
);
