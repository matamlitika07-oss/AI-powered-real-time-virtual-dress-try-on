import { Camera, Check, Settings, Download, CameraOff } from 'lucide-react';
import { useTryOn } from '@/hooks/useTryOn';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

const GARMENTS = [
  { id: 'shirt', name: 'White Button-Down', src: '/clothes/shirt.png' },
  { id: 'blazer', name: 'Navy Blazer', src: '/clothes/blazer.png' },
  { id: 'hoodie', name: 'Grey Hoodie', src: '/clothes/hoodie.png' },
  { id: 'striped', name: 'Striped Long Sleeve', src: '/clothes/striped.png' },
  { id: 'tshirt', name: 'Black T-Shirt', src: '/clothes/tshirt.png' },
  { id: 'jacket', name: 'Brown Leather Jacket', src: '/clothes/jacket.png' }
];

export default function TryOn() {
  const { toast } = useToast();
  const {
    videoRef,
    canvasRef,
    selectedGarment,
    setSelectedGarment,
    opacity,
    setOpacity,
    poseStatus,
    fps,
    webcamError,
    captureLook
  } = useTryOn();

  const handleCapture = () => {
    if (!selectedGarment) {
      toast({
        title: "No garment selected",
        description: "Please select a garment from the catalog before capturing your look.",
        variant: "destructive"
      });
      return;
    }
    captureLook();
    toast({
      title: "Look Captured",
      description: "Your virtual try-on look has been saved.",
    });
  };

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-background text-foreground uppercase tracking-widest font-sans">
      {/* LEFT SIDE - TRY-ON PREVIEW (60%) */}
      <div className="relative w-full lg:w-[60%] flex-none h-[60vh] lg:h-screen bg-black overflow-hidden flex items-center justify-center border-b lg:border-b-0 lg:border-r border-border">
        {/* Hidden Video Feed */}
        <video 
          ref={videoRef} 
          className="hidden" 
          playsInline 
          muted 
        />
        
        {webcamError ? (
          <div className="flex flex-col items-center justify-center space-y-4 text-center p-6 text-muted-foreground">
            <CameraOff className="w-12 h-12 mb-2 opacity-50" />
            <p className="text-sm tracking-widest max-w-xs">{webcamError}</p>
          </div>
        ) : (
          <>
            <canvas 
              ref={canvasRef} 
              className="w-full h-full object-contain pointer-events-none" 
            />
            
            {/* FPS Counter */}
            <div className="absolute top-4 left-4 font-mono text-xs opacity-50 px-2 py-1 bg-black/50 rounded backdrop-blur-sm border border-white/10">
              {fps} FPS
            </div>
            
            {/* Pose Status Overlay */}
            <div className="absolute bottom-24 left-0 right-0 flex justify-center pointer-events-none">
              <div className="px-4 py-2 bg-black/80 backdrop-blur-md text-xs tracking-[0.2em] rounded-full border border-white/10 text-primary uppercase shadow-xl flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                {poseStatus}
              </div>
            </div>
            
            {/* Capture Button */}
            <div className="absolute bottom-8 left-0 right-0 flex justify-center">
              <Button 
                onClick={handleCapture}
                size="lg"
                className="bg-primary hover:bg-primary/90 text-primary-foreground font-bold tracking-widest px-8 rounded-full shadow-2xl transition-all active:scale-95"
              >
                <Camera className="mr-2 w-5 h-5" />
                Capture Look
              </Button>
            </div>
          </>
        )}
      </div>

      {/* RIGHT SIDE - CATALOG & CONTROLS (40%) */}
      <div className="w-full lg:w-[40%] flex flex-col h-[40vh] lg:h-screen bg-sidebar">
        
        {/* Header */}
        <header className="p-6 border-b border-border flex items-center justify-between shrink-0">
          <div>
            <h1 className="text-xl font-bold tracking-[0.3em] text-white">ATELIER</h1>
            <p className="text-[10px] text-muted-foreground mt-1">Virtual Fitting Room</p>
          </div>
        </header>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-8 custom-scrollbar">
          
          {/* Controls */}
          <div className="space-y-4">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span className="flex items-center gap-2">
                <Settings className="w-4 h-4" /> Garment Opacity
              </span>
              <span>{opacity}%</span>
            </div>
            <Slider 
              value={[opacity]} 
              min={0} 
              max={100} 
              step={1} 
              onValueChange={(val) => setOpacity(val[0])}
              className="w-full"
            />
          </div>

          <div className="w-full h-px bg-border" />

          {/* Catalog */}
          <div>
            <h2 className="text-sm font-semibold tracking-widest mb-6 flex items-center gap-2">
              Collection
              <span className="text-[10px] bg-primary/20 text-primary px-2 py-0.5 rounded-full">
                {GARMENTS.length} Items
              </span>
            </h2>
            
            <div className="grid grid-cols-2 gap-4">
              {GARMENTS.map((garment) => {
                const isSelected = selectedGarment === garment.src;
                
                return (
                  <button
                    key={garment.id}
                    onClick={() => setSelectedGarment(garment.src)}
                    className={cn(
                      "group relative flex flex-col text-left transition-all duration-300 outline-none rounded-md overflow-hidden bg-card border-2",
                      isSelected 
                        ? "border-primary shadow-[0_0_20px_rgba(212,175,55,0.15)] scale-[1.02]" 
                        : "border-transparent hover:border-border"
                    )}
                  >
                    <div className="aspect-[4/5] bg-black/50 w-full relative p-4 flex items-center justify-center overflow-hidden">
                      {isSelected && (
                        <div className="absolute top-2 right-2 z-10 w-6 h-6 rounded-full bg-primary flex items-center justify-center shadow-lg">
                          <Check className="w-4 h-4 text-primary-foreground" />
                        </div>
                      )}
                      
                      {/* Decorative backdrop for images */}
                      <div className="absolute inset-0 bg-gradient-to-tr from-black/80 via-transparent to-white/5 opacity-50" />
                      
                      <img 
                        src={garment.src} 
                        alt={garment.name}
                        className={cn(
                          "w-full h-full object-contain relative z-0 transition-transform duration-500",
                          isSelected ? "scale-110" : "group-hover:scale-110"
                        )}
                        crossOrigin="anonymous"
                      />
                    </div>
                    
                    <div className="p-3 border-t border-border/50 bg-card/80 backdrop-blur-sm">
                      <p className="text-[10px] font-medium tracking-widest text-white truncate w-full">
                        {garment.name}
                      </p>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
