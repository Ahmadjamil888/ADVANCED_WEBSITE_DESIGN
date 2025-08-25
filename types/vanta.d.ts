declare module 'vanta/dist/vanta.net.min.js' {
  interface VantaEffect {
    destroy(): void;
  }

  interface VantaOptions {
    el: HTMLElement;
    THREE: any;
    mouseControls?: boolean;
    touchControls?: boolean;
    gyroControls?: boolean;
    minHeight?: number;
    minWidth?: number;
    scale?: number;
    scaleMobile?: number;
    color?: number;
    backgroundColor?: number;
    points?: number;
    maxDistance?: number;
    spacing?: number;
  }

  function vanta(options: VantaOptions): VantaEffect;
  export default vanta;
}