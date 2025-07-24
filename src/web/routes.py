from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, Any

from src.core.config import config_manager, Config

router = APIRouter()
templates = Jinja2Templates(directory="src/web/templates")

@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main configuration page."""
    profiles = config_manager.get_all_profiles()
    active_profile_name = config_manager.get_active_profile_name()
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "config": config_manager.config,
        "profiles": profiles,
        "active_profile_name": active_profile_name,
        "config_manager": config_manager,  # Pass the manager to the template
        "message": ""
    })

class ProfileUpdateData(BaseModel):
    original_name: str
    new_name: str
    data: Dict[str, Any]

@router.post("/api/config/update")
async def update_config(profile: ProfileUpdateData):
    """Updates an existing profile's data and optionally renames it."""
    try:
        # First, save the data to the original profile name
        config_manager.save_profile(profile.original_name, profile.data)

        # If the name has changed, rename the profile
        if profile.original_name != profile.new_name:
            config_manager.rename_profile(profile.original_name, profile.new_name)
        
        return {"status": "success", "message": f"Profile '{profile.new_name}' updated."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class NewProfileRequest(BaseModel):
    profile_name: str

@router.post("/api/config/new")
async def new_config(req: NewProfileRequest):
    """Creates a new, empty profile."""
    try:
        if not req.profile_name.strip():
            raise ValueError("Profile name cannot be empty.")
        if req.profile_name in config_manager.get_all_profiles():
            raise ValueError(f"Profile '{req.profile_name}' already exists.")
            
        # Create a new profile with default values
        config_manager.save_profile(req.profile_name, {})
        return {"status": "success", "message": f"Profile '{req.profile_name}' created."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/api/config/{profile_name}")
async def get_profile_data(profile_name: str):
    """Gets the data for a specific profile."""
    profiles = config_manager.get_all_profiles()
    profile_data = profiles.get(profile_name)
    if not profile_data:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_name}' not found.")
    return profile_data

class ActivateProfileRequest(BaseModel):
    profile_name: str

@router.post("/api/config/activate")
async def activate_config(req: ActivateProfileRequest):
    """Activates a configuration profile."""
    try:
        config_manager.activate_profile(req.profile_name)
        # Redirect to the root to show the updated config
        return RedirectResponse("/", status_code=303)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class DeleteProfileRequest(BaseModel):
    profile_name: str

@router.post("/api/config/delete")
async def delete_config(req: DeleteProfileRequest):
    """Deletes a configuration profile."""
    try:
        config_manager.delete_profile(req.profile_name)
        return {"status": "success", "message": f"Profile '{req.profile_name}' deleted."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))